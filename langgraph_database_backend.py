from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- Tools ---------------- #

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform arithmetic operations.
    Supported operations:
    add, sub, mul, div
    """

    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero"}
        result = first_num / second_num
    else:
        return {"error": "Invalid operation"}

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price from Alpha Vantage.
    """

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey=X3XTUXI8K2X9QUH1"
    )

    response = requests.get(url)
    return response.json()


tools = [
    search_tool,
    calculator,
    get_stock_price,
]

# Bind tools
llm_with_tools = llm.bind_tools(tools)

# ---------------- State ---------------- #

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------- Chat Node ---------------- #

def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response]
    }


# ---------------- Tool Node ---------------- #

tool_node = ToolNode(tools)

# ---------------- SQLite Memory ---------------- #

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn)

# ---------------- Graph ---------------- #

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ---------------- Thread History ---------------- #

def retrieve_all_threads():
    threads = set()

    for checkpoint in checkpointer.list(None):
        threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(threads)