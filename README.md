# RAG-Paper-Chat-v1
v1 is a semantic RAG build designed as resource and playbook for AI system project recipes, patterns, and shapes.

- (add diagram here)

**RAG (retrieval-augmented generation)** with LangChain = find relevant text first, then let the model answer from it.

### How it works
Semantic RAG retrieves passages by *meaning* (embedding similarity), not keywords.
It runs as two pipelines:

1. **Indexing** (once per document): load document → split into chunks →
   embed each chunk → store chunks + embeddings in a vector DB
2. **Querying** (every question): embed the question → retrieve the most similar
   chunks → send question + chunks to the LLM → generate a cited answer


## App 
This app helps self-taught learners understand ML, DL and LLM papers by simplifying them, answering follow-up questions grounded in the paper, and building a learning plan


## Phase 0: Setup Project
```bash
uv init --package app-name-here --python 3.12
cd app-name && git init
mkdir -p tests data/raw && touch tests/__init__.py data/raw/.gitkeep
uv add --dev pytest ruff
```

## Phase 1: Design Schemas(Pydantic)
```bash
uv add pydantic 
```

### The idea behind RAG schemas
Data in a RAG app comes in three kinds. Each kind usually needs its own schema:

1. What you store: the paper, plus the chunks you split it into and retrieve later.
2. What moves through a chat: messages, each with a role (user or assistant) and content.
3. What the LLM returns: structured output for each of your 4 tasks.

The simplest setup that works is roughly one small model per item in those lists. Getting it right is mostly about choosing the fields.

- Follow the data through the app. A RAG app does three things: it ingests a paper, retrieves pieces of it, and generates a response. Each stage handles a different kind of thing, and each kind gets a model. Ingesting gives you Paper, retrieving gives you Chunk, and generating gives you the output models



#### Phase 1(a): Test Schemas 
`test/test_schema.py`
**Test**:
- Paper Schema 
- Chunk Schema
- Ciatation Schema
- Answer Schema
- PlanStep Schema
- LearningPlan Schema
run `uv run pytest -v`

## Phase 2: ingest data
 1. Load: documents to search

 add packages:

```bash
uv add pypdf langchain-core langchain-text-splitters
```
A Document is text (page_content) plus metadata (where it came from).
Quick manual check (real PDFs)
```python
uv run python -c "
from paper_chat.ingest import load_pdfs
papers, docs = load_pdfs()
for p in papers: print(p.model_dump())
print(len(docs), 'pages')
print(docs[0].metadata)
"
```

### Build store
```python
uv run python -m paper_chat.ingest
```


## Phase 3: Split Docs into chunks
Small chunks keep retrieval precise. Overlap keeps sentences from being cut in half.

**RecursiveCharacterTextSplitter** tries a list of separators, in order: "\n\n" → "\n" → " " → "". If a piece is still too big, it moves on to the next, smaller separator. So it always reaches your size limit, while splitting at the most natural boundary available: paragraphs first, then lines, then words.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)
print(f"{len(docs)} documents -> {len(chunks)} chunks")
```

## Phase 4: Embed, Store, Retrieve, Generate
```bash
uv add langchain-openai langchain-chroma python-dotenv
```












| Package | Used for |
|---|---|
| `pydantic` | Schemas (`Paper`, `Chunk`, `Answer`, ...) |
| `pypdf` | Reading PDF text (BSD license) |
| `langchain-core` | `Document`, prompts, output parsers, runnables |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `langchain-openai` | `ChatOpenAI`, `OpenAIEmbeddings` (needs `OPENAI_API_KEY`) |
| `langchain-chroma` | Chroma vector store (installs `chromadb`) |
| `python-dotenv` | Loads `.env`; the import name is `dotenv` |
| `pytest`, `ruff` (dev) | Tests, linting/formatting |
