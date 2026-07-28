import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# *********************************** Utility Functions ******************************

def generate_thread_id():
    return uuid.uuid4()

def get_chat_title(thread_id):

    conversation = load_conversation(thread_id)

    for msg in conversation:

        if msg["role"] == "user":

            title = msg["content"].strip()

            if len(title) > 35:
                title = title[:35] + "..."

            return title

    return "New Chat"


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

# ******************************** Sidebar ******************************** #

st.sidebar.title("📄 LangGraph PDF ChatBot")

# Current thread
st.sidebar.markdown(f"**Thread ID:** `{st.session_state['thread_id']}`")

# ---------------- New Chat ---------------- #

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# ---------------- PDF Upload ---------------- #

metadata = thread_document_metadata(str(st.session_state["thread_id"]))

if metadata:
    st.sidebar.success(
        f"""📄 **{metadata.get('filename')}**

Pages: {metadata.get('documents')}

Chunks: {metadata.get('chunks')}
"""
    )
else:
    st.sidebar.info("No PDF uploaded for this chat.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

if uploaded_pdf:

    with st.sidebar.status(
        "Indexing PDF...",
        expanded=True,
    ) as status:

        summary = ingest_pdf(
            uploaded_pdf.getvalue(),
            thread_id=str(st.session_state["thread_id"]),
            filename=uploaded_pdf.name,
        )

        status.update(
            label="✅ PDF Indexed",
            state="complete",
            expanded=False,
        )

    st.sidebar.success(
        f"Indexed **{summary['filename']}**"
    )

st.sidebar.divider()

# ---------------- Previous Chats ---------------- #

st.sidebar.subheader("💬 Previous Chats")

threads = retrieve_all_threads()

threads = list(dict.fromkeys(threads))

threads.reverse()

for thread in threads:

    title = get_chat_title(thread)

    if st.sidebar.button(
        title,
        key=f"thread-{thread}",
        use_container_width=True,
    ):

        st.session_state["thread_id"] = thread
        st.session_state["message_history"] = load_conversation(thread)

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