# LangGraph ChatBot

A small conversational AI application built with LangGraph, Groq, and Streamlit. The app keeps conversation state in a LangGraph in-memory checkpointer and presents a browser-based chat interface.

## Features

- Streamlit chat interface
- LangGraph workflow with a single LLM chat node
- Conversation history persisted for the active Streamlit session
- Groq-hosted `llama-3.3-70b-versatile` model

## Project structure

```
.
├── langgraph_backend.py   # LangGraph workflow and Groq model setup
├── streamlit_frontend.py  # Streamlit chat application
├── requirements.txt       # Python dependencies
└── .env.example           # Environment-variable template
```

## Prerequisites

- Python 3.10 or later
- A [Groq API key](https://console.groq.com/keys)

## Setup

1. Clone the repository and enter its directory.

   ```bash
   git clone <repository-url>
   cd LangGraphChatBot
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, use `.venv\\Scripts\\Activate.ps1` instead.

3. Install the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Create a local environment file and add your key.

   ```bash
   cp .env.example .env
   ```

   Set `GROQ_API_KEY` in `.env` to your Groq API key. Never commit this file.

## Run the app

```bash
streamlit run streamlit_frontend.py
```

Streamlit will print the local URL, typically `http://localhost:8501`.

## How it works

The frontend sends each user message to the compiled LangGraph workflow. Its `chat_node` invokes the Groq language model and returns the assistant response. An `InMemorySaver` checkpointer associates messages with the configured thread ID for the running process.

## Notes

- The current thread ID is `thread-1`; restarting the app clears the in-memory LangGraph checkpoint.
- `.env` and virtual environments are intentionally ignored by Git.

## License

No license has been specified. Add a `LICENSE` file before distributing or reusing this project publicly.
