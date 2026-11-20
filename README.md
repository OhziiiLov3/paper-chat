# Paper-Chat

## App 
This app helps self-taught learners understand ML, DL and LLM papers by simplifying them, answering follow-up questions grounded in the paper, and building a learning pl

## Phase 0: Setup Project
```bash
uv init --package app-name-here --python 3.12
cd app-name && git init
mkdir -p tests data/raw && touch tests/__init__.py data/raw/.gitkeep
uv add --dev pytest ruff
```

# Phase 1: Schemas(Pydantic)
```bash
uv add pydantic 
```

# The idea behind RAG schemas
Data in a RAG app comes in three kinds. Each kind usually needs its own schema:

1. What you store: the paper, plus the chunks you split it into and retrieve later.
2. What moves through a chat: messages, each with a role (user or assistant) and content.
3. What the LLM returns: structured output for each of your 4 tasks.

The simplest setup that works is roughly one small model per item in those lists. Getting it right is mostly about choosing the fields.

- Follow the data through the app. A RAG app does three things: it ingests a paper, retrieves pieces of it, and generates a response. Each stage handles a different kind of thing, and each kind gets a model. Ingesting gives you Paper, retrieving gives you Chunk, and generating gives you the output models

