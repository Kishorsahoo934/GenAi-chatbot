import streamlit as st
from langgraph.graph import StateGraph, END, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import os

# Load environment variables
load_dotenv()

# Connect to SQLite database
conn = sqlite3.connect("chatbot.sqlite3", check_same_thread=False)
memory = SqliteSaver(conn)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.1
)

# Define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define LLM node
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

# Build LangGraph
graph = StateGraph(ChatState)
graph.add_node("chatnode", chat_node)
graph.set_entry_point("chatnode")
graph.add_edge("chatnode", END)
chatbot = graph.compile(checkpointer=memory)

# Streamlit UI setup
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("🧠 Chatbot with Memory")

st.sidebar.title("⚖️ Chat History")
thread_id = st.sidebar.text_input("Session ID", value="kishor_session_1")
config = {"configurable": {"thread_id": thread_id}}

# Delete history button
if st.sidebar.button("🗑️ Delete History"):
    try:
        # Delete the history from the database
        memory.delete_thread(config)
        # Commit the changes to the database file
        conn.commit()
        st.sidebar.success(f"History for session ID '{thread_id}' deleted successfully.")
        # Rerun the app to refresh the display
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error deleting history: {e}")

# Retrieve past conversation
try:
    state = chatbot.get_state(config)
    history = state.values.get("messages", [])
except Exception:
    history = []

# Display chat history in UI
for msg in history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run graph with input
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    response = chatbot.invoke(initial_state, config=config)
    bot_reply = response["messages"][-1].content

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
