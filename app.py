import streamlit as st
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda


# ------------------ ENV SETUP ------------------
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ------------------ STREAMLIT UI ------------------
st.title("Conversational RAG with PDF")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API Key", type="password")

if not api_key:
    st.warning("Please enter the Groq API key")
    st.stop()

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant"
)

session_id = st.text_input("Session ID", value="default")

if "store" not in st.session_state:
    st.session_state.store = {}

# ------------------ PDF UPLOAD ------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

documents = []

for i, uploaded_file in enumerate(uploaded_files):
    temp_path = f"./temp_{i}.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_path)
    documents.extend(loader.load())

# ------------------ SPLIT + VECTORSTORE ------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=500
)

splits = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

# ------------------ CONTEXTUALIZE QUESTION ------------------
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question, "
            "rewrite the question so it can be understood without chat history."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

contextualize_chain = contextualize_prompt | llm

# ------------------ RETRIEVER RUNNABLE ------------------
def retrieve_documents(inputs):
    standalone_question = contextualize_chain.invoke(inputs).content
    docs = retriever.invoke(standalone_question)
    return {
        "context": docs,
        "input": inputs["input"],
        "chat_history": inputs["chat_history"],
    }

retrieval_runnable = RunnableLambda(retrieve_documents)

# ------------------ QA PROMPT ------------------
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a question-answering assistant. "
            "Use the provided context to answer the question. "
            "If you don't know, say you don't know. "
            "Use at most three sentences.\n\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

qa_chain = qa_prompt | llm

# ------------------ FINAL RAG PIPELINE ------------------
rag_chain = retrieval_runnable | qa_chain

# ------------------ CHAT HISTORY STORE ------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ------------------ CHAT UI ------------------
user_input = st.text_input("Your question")

if user_input:
    response = conversational_rag.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    st.write("### Assistant")
    st.write(response.content)

    # st.write("### Chat History")
    # st.write(st.session_state.store[session_id].messages)
