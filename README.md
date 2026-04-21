Here's a complete `README.md` for your project:

```markdown
# RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot with user authentication, built with FastAPI. Each user gets their own private document library and chat history — upload PDFs, DOCX, or HTML files and ask questions about them.

## Features

- **User accounts** — signup/login with bcrypt-hashed passwords and signed session cookies
- **Per-user document isolation** — you only see and retrieve from your own documents
- **RAG with chat history** — LangGraph pipeline contextualizes follow-up questions using conversation history
- **Multiple LLM choices** — switch between Llama 3.3 70B, Llama 3.1 8B, and GPT-OSS 120B via Groq
- **Local embeddings** — FastEmbed (ONNX) runs embeddings on CPU without API calls or torch
- **Document management UI** — upload, list, and delete documents from a simple web page

## Tech stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI + Jinja2 templates |
| LLM | Groq (Llama 3.3 70B by default) |
| Embeddings | FastEmbed with `BAAI/bge-small-en-v1.5` (local, 384-dim) |
| Vector store | Chroma (persistent, local) |
| Database | Postgres via Neon + SQLAlchemy ORM |
| RAG orchestration | LangChain + LangGraph |
| Auth | Starlette `SessionMiddleware` + bcrypt |

## Project structure

```
rag-fastapi-project/
├── main.py                 # FastAPI app, routes, auth, templates/static mounts
├── chroma_utils.py         # Chroma vector store + FastEmbed embeddings
├── db_utils.py             # SQLAlchemy models & CRUD for Neon Postgres
├── langchain_utils.py      # Groq LLM + LangGraph RAG pipeline
├── pydantic_models.py      # Request/response schemas
├── requirements.txt
├── render.yaml             # Render deployment config
├── .env.example
├── templates/
│   ├── base.html
│   ├── index.html          # Chat page
│   ├── documents.html      # Document management
│   ├── login.html
│   └── signup.html
└── static/
    ├── css/style.css
    └── js/
        ├── chat.js
        └── documents.js
```

## How it works

1. User signs up → bcrypt-hashed password stored in Postgres → signed session cookie issued.
2. User uploads a PDF/DOCX/HTML → file chunked (1000 chars with 200 overlap) → each chunk embedded locally with FastEmbed → stored in Chroma with `user_id` metadata.
3. User asks a question → LangGraph pipeline runs:
   - **Contextualize:** if there's chat history, Groq rewrites the question into a standalone form.
   - **Retrieve:** Chroma searches only chunks matching this user's `user_id`, returning top 2.
   - **Generate:** Groq answers using retrieved chunks + chat history as context.
4. Chat turn persisted to Postgres, session stays alive across messages.

## Local setup

### 1. Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/keys) (free)
- A [Neon Postgres database](https://neon.tech) (free tier)

### 2. Clone and install

```bash
git clone <your-repo-url>
cd rag-fastapi-project
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
GROQ_API_KEY=gsk_your_key_here
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require
SESSION_SECRET=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
```

### 4. Run

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/` — you'll be redirected to the login page. Click **Sign up** to create an account.

**First run note:** FastEmbed downloads the embedding model (~130 MB) on first use. Takes about 20 seconds. Cached afterward in `./fastembed_cache/`.

### 5. Available endpoints

**Pages:**
- `/signup`, `/login`, `/logout`
- `/` — chat UI (auth required)
- `/docs-ui` — upload & manage documents (auth required)

**JSON API** (all require authentication via session cookie):
- `POST /chat` — send a question, get an answer
- `POST /upload-doc` — upload and index a file
- `GET /list-docs` — list your documents
- `POST /delete-doc` — delete one of your documents


Chroma data and the FastEmbed model cache persist at `/var/data` on a 1 GB disk, so restarts don't lose data or trigger a re-download.

## Credits

Project structure inspired by [this FastAPI RAG tutorial](https://blog.futuresmart.ai/building-a-production-ready-rag-chatbot-with-fastapi-and-langchain), adapted with Groq, FastEmbed, LangGraph, authentication, and Render deployment.
```
