import streamlit as st
import uuid

from langgraph_database_backend import chatbot, retrieve_all_threads

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)

# *********************************** Utility Functions ******************************

def generate_thread_id():
    return uuid.uuid4()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        st.session_state["chat_titles"][thread_id] = "New Chat"


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    messages = state.values.get("messages", [])

    conversation = []

    for msg in messages:

        # Ignore raw tool output
        if isinstance(msg, ToolMessage):
            continue

        # Ignore AI messages that only contain tool calls
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue

        if isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "assistant"

        conversation.append(
            {
                "role": role,
                "content": msg.content,
            }
        )

    return conversation


# ******************************** Session State *********************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

add_thread(st.session_state["thread_id"])


# ******************************** Sidebar *********************************

st.sidebar.title("LangGraph ChatBot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:

    title = st.session_state["chat_titles"].get(
        thread_id,
        "New Chat",
    )

    if st.sidebar.button(title, key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)
        st.rerun()


# ******************************** Main UI *********************************

st.title("💬 LangGraph ChatBot")

# Display previous messages

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Type your message...")


if user_input:

    # ---------------- Conversation Title ---------------- #

    if (
        st.session_state["chat_titles"][
            st.session_state["thread_id"]
        ]
        == "New Chat"
    ):

        title = user_input.strip()

        if len(title) > 30:
            title = title[:30] + "..."

        st.session_state["chat_titles"][
            st.session_state["thread_id"]
        ] = title

    # ---------------- User Message ---------------- #

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # ---------------- Assistant ---------------- #

    def assistant_response():

        final_answer = ""

        for message, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):

            # Skip tool output
            if isinstance(message, ToolMessage):
                continue

            # Skip tool-calling AI message
            if isinstance(message, AIMessage) and message.tool_calls:
                continue

            if message.content:
                final_answer += message.content
                yield message.content

    with st.chat_message("assistant"):
        ai_response = st.write_stream(assistant_response())

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_response,
        }
    )