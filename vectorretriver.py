from langchain_core.documents import Document
from langchain_chroma import Chroma
import os 
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


llm = ChatGroq(model="lllama-3.1-8b-instant", groq_api_key=groq_api_key)

documnets = [
    Document(
        page_content="Dogs are great companion, Known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-docs"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-docs"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "mammal-pets-docs"},
    ),
    Document(
        page_content="Goldfish are popular  pets for begninners, requiring relatively simple care.",
        metadata={"source": "mammal-pets-docs"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-docs"},
    )
]
# print(documnets)

# embedding techinque
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# creating vectordb by embedding the documents
vectorstore = Chroma.from_documents(documnets, embedding=embeddings)

print(vectorstore)




