from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# groq-llama2-70b-chat
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key,)
print(model)

# print(model.invoke([HumanMessage(content="Hello, my name is sahil and I am a Chief AI Engineer")]))

model.invoke(
    [
        HumanMessage(content="Hello, my name is sahil and I am a Chief AI Engineer"),
        AIMessage(content="Hello Sahil, nice to meet you. As a Chief AI Engineer, you must be working on some exciting projects, leveraging the power of artificial intelligence to drive innovation and solve complex problems. What specific areas of AI are you focused on, such as machine learning, natural language processing, or computer vision? I'd love to hear more about your work."),
        HumanMessage(content="Hey what's my name and what do I do?")
    ]
)


store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model,get_session_history)

config = {"configurable":{"session_id":"chat1"}}

# response = with_message_history.invoke(
#     [HumanMessage(content="Hi, my name is sahil and I am a Chief AI Engineer")],
#     config=config)

# print("AI: ",response.content)

response = with_message_history.invoke(
    [HumanMessage(content="what's my name and what do I do?")],
    config=config)

# print("AI: ",response.content)


## prompt templates
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Answer the question to the best of your ability."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

res = chain.invoke({'messages': [HumanMessage(content="Hi my name is sahil")]})

print(res)



