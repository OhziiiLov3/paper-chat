# Ingestion: Turning PDFs into a Searchable Vector Store

**Folder:** `src/paper_chat/` (`ingest.py`)

## What you'll learn
- How the indexing half of a RAG app works: load → split → embed → store
- How to load PDFs into LangChain `Document`s without the sunset `langchain-community` package
- How chunk size, overlap, and the recursive splitter shape what retrieval can find
- How stable chunk ids prevent duplicates and make citations possible
- How to structure a pipeline as importable functions, so the query side can reuse it

This file runs once per set of documents. It reads your PDFs, cuts them into small passages, turns each passage into numbers that capture its meaning, and saves them so the chatbot can search them later.

---

## 1. Two pipelines, one file each

A RAG app (retrieval-augmented generation: the LLM answers using passages fetched from your documents) has two separate pipelines:

| Pipeline | When it runs | Steps | File |
|---|---|---|---|
| **Indexing** | Once per document set | Load → split → embed → store | `ingest.py` |
| **Querying** | Every user question | Embed question → retrieve → generate answer | `rag.py` |

Indexing is slow and costs money (every chunk is embedded). Querying must be fast. Keeping them in separate files lets you change one without touching the other.

The module docstring states the job in one line. It only counts as the module's docstring if it is the **first line** of the file:

```python
"""Indexing pipeline: load PDFs -> split into chunks -> embed -> store in Chroma."""
```

> **Key point:** "Ingest" means the indexing pipeline only. Anything the user's question touches (the LLM, prompts) belongs in `rag.py`.

---

## 2. Paths and settings as constants

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "llm_academic_papers"
EMBEDDING_MODEL = "text-embedding-3-small"
```

`parents[N]` counts up from the file's folder:

| Expression | Folder |
|---|---|
| `parents[0]` | `paper-chat/src/paper_chat` |
| `parents[1]` | `paper-chat/src` |
| `parents[2]` | `paper-chat` (project root) |

- **`.resolve()`** makes the path absolute, so it doesn't depend on which folder you run the command from.
- **`data/raw/`** holds original inputs that you never edit. It is git-ignored, so client PDFs don't end up in the repo.
- **`.chroma/`** is the vector database folder. It is also git-ignored, because it can be rebuilt from `raw/` at any time.
- **`EMBEDDING_MODEL`** is defined once because ingest and query **must** use the same embedding model. Vectors from two different models can't be compared.

---

## 3. Load: one `Document` per page

A **`Document`** (from `langchain_core`) is a container with two parts: `page_content` (the text) and `metadata` (a dict of facts about the text). It doesn't read files. `pypdf` reads the PDF, and you put each page's text into a `Document`:

```python
def load_pdfs(raw_dir: Path = RAW_DIR) -> list[Document]:
    docs = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages):
            docs.append(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={
                        "paper_id": pdf_path.stem,
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )
    return docs
```

Real result on the 3 project papers: **92 pages, 0 empty pages.** First page's metadata:

```
{'paper_id': '2505.17120v2', 'source': '2505.17120v2.pdf', 'page': 0}
```

| Detail | Why |
|---|---|
| `paper_id = pdf_path.stem` | The filename without `.pdf`. It is **stable**: re-ingesting the same file gives the same id, so you don't get duplicates. |
| `or ""` | `extract_text()` can return nothing for image-only (scanned) pages. |
| `raw_dir` parameter with a default | Tests can point it at a folder of small sample PDFs. |
| `sorted(...)` | Same order every run, so chunk numbering is repeatable. |

A `Document` is **temporary packaging**, not your schema. One `Document` is one page; one `Paper` (see [schema.py.md](schema.py.md)) is one whole paper. The metadata only needs to carry what you'll need later, most importantly `paper_id`, which links each chunk back to its `Paper`.

**Why not `PyPDFLoader`?** It lives in `langchain-community`, which LangChain has sunset. Its repository was archived in 2026. Building the `Document` yourself is a few lines and has one less dependency to go stale.

**Library choice for client work (licenses checked on PyPI; get legal review for real contracts):**

| Library | License | Notes |
|---|---|---|
| `pypdf` | BSD-3 | Light, plain text only. Tables lose their column structure. |
| Docling (`langchain-docling`) | MIT | Understands tables, headings, and columns. Heavy (~85 packages). Not tested in this project. |
| PyMuPDF | AGPL-3.0 | Good quality, but AGPL can require open-sourcing the client's app. |

---

## 4. Split: chunks small enough to retrieve precisely

A **chunk** is a small passage of text. Retrieval searches chunks, not whole papers, because a focused passage matches a question better and fits in the LLM's prompt.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
```

Real result: **92 pages → 348 chunks**, sizes from 132 to 1000 characters.

| Parameter | Meaning | Trade-off |
|---|---|---|
| `chunk_size=1000` | Maximum characters per chunk | Too small (e.g. 200) gives one or two sentences, with too little context. Too big makes retrieval less precise. |
| `chunk_overlap=150` | Up to this many characters repeated between neighboring chunks | Keeps a sentence from being cut off with no context on either side. |

The overlap is a **maximum**, not an exact amount. The splitter only overlaps on clean boundaries (spaces, newlines). Between the first two chunks, the real overlap was 80 characters.

**Why the *recursive* splitter?** It tries separators in order (`"\n\n"` → `"\n"` → `" "` → `""`) until every piece fits. The simpler `CharacterTextSplitter` uses only one separator (`"\n\n"`). `pypdf` text has no blank lines, so on the first 3 pages of one paper (real output):

```
CharacterTextSplitter:          1 chunk,   size 13,124   ← ignored chunk_size
RecursiveCharacterTextSplitter: 16 chunks, sizes 154–992
```

Each page is split separately, so **a chunk never spans two pages**. That keeps the `page` metadata accurate for citations.

> **Key point:** Chunk size decides what retrieval *can* find. If an answer is split across two chunks, or a table loses its header, no prompt can fix it later.

---

## 5. Stable chunk ids: numbering within each paper

Each chunk inherits its page's metadata. The loop adds two more fields, `chunk_index` and `chunk_id`, which match `Chunk.index` and `Chunk.id` in the schema:

```python
counts: dict[str, int] = {}
for chunk in chunks:
    paper_id = chunk.metadata["paper_id"]
    index = counts.get(paper_id, 0)  # chunks already numbered for this paper
    counts[paper_id] = index + 1  # the next one gets the next number
    chunk.metadata["chunk_index"] = index
    chunk.metadata["chunk_id"] = f"{paper_id}-c{index}"
```

- `counts` tracks how many chunks each paper has so far. `.get(paper_id, 0)` returns 0 for a paper's first chunk.
- That count **is** the new chunk's 0-based index. Add 1 in one place only (the update).

Real result: chunks per paper `{'2605.13339v4': 154, '2604.17031v3': 118, '2505.17120v2': 76}`, each numbered 0, 1, 2… with no gaps. All 348 ids are unique. Second chunk:

```
{'paper_id': '2505.17120v2', 'source': '2505.17120v2.pdf', 'page': 0,
 'chunk_index': 1, 'chunk_id': '2505.17120v2-c1'}
```

Short, readable ids like `2505.17120v2-c1` are what the LLM will copy into a `Citation.chunk_id` later.

---

## 6. Embed and store: build once, load many times

An **embedding** is a list of numbers that represents the meaning of a text. Texts with similar meaning get similar numbers, which is why this kind of search is called *semantic*. A **vector store** (here, Chroma) saves those numbers and finds the closest ones to a question.

```python
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(chunks: list[Document]) -> Chroma:
    return Chroma.from_documents(
        chunks,
        get_embeddings(),
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )


def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
```

| Function | Cost | Used by |
|---|---|---|
| `build_vectorstore` | Paid: embeds every chunk | `main()`, once per ingest |
| `load_vectorstore` | Free: opens what's on disk | `rag.py`, every run |

**Why `ids=` matters.** With ids, `langchain-chroma` calls Chroma's `upsert` (checked in the installed source). Re-running ingestion *replaces* chunks with the same id. Without ids, every run **adds another copy** of every chunk, and retrieval starts returning duplicates.

Real check of the local store (read directly from disk with `chromadb`, no API call):

```
llm_academic_papers 348 ['2505.17120v2-c0', '2505.17120v2-c1', '2505.17120v2-c2']
```

348 stored = 348 chunks, so there are no duplicates.

Upsert won't delete ids that no longer exist. If you change chunk size or numbering, delete `.chroma/` and ingest again.

**Intel Macs:** `chromadb` needs `onnxruntime`, whose newest versions have no Intel Mac builds. The project's `pyproject.toml` has a `[tool.uv] required-environments` setting so uv picks a compatible version there.

---

## 7. Retrieve and sanity-check

```python
def get_retriever(k: int = 3):
    return load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": k})


def check_retrieval(query: str = "What is the main contribution of the paper?", k: int = 3) -> None:
    retriever = get_retriever(k=k)
    print(f"\nQuery: {query}")
    for doc in retriever.invoke(query):
        print(f"- {doc.metadata['chunk_id']} (page {doc.metadata['page']})")
        print(f"  {doc.page_content[:200]!r}")
```

- A **retriever** takes a question and returns the `k` most similar chunks. `rag.py` imports `get_retriever` and never touches PDFs.
- `check_retrieval` is a quick eyeball test. Printing `chunk_id` shows problems quickly: the same id twice means duplicates, and skipped numbers mean the counter is wrong.
- Each check embeds one short query, which is a small paid call.

To check retrieval without re-ingesting:

```
uv run python -c "from paper_chat.ingest import check_retrieval; check_retrieval()"
```

---

## 8. Functions plus `main()`: importable, not automatic

```python
def main() -> None:
    docs = load_pdfs()
    chunks = split_docs(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    build_vectorstore(chunks)
    print(f"Stored in {CHROMA_DIR}")
    check_retrieval()


if __name__ == "__main__":
    main()
```

`if __name__ == "__main__":` is true only when you run the file directly (`uv run python -m paper_chat.ingest`), not when another file imports it. So `from paper_chat.ingest import get_retriever` in `rag.py` doesn't re-read PDFs or re-embed anything.

Code at the top level of a module runs on **every** import. Before this file was refactored, importing it would have re-embedded all chunks. Small, separate functions are also what make each step **testable**, as with the schemas in [schema.py.md](schema.py.md).

---

## Key ideas
- RAG is two pipelines: indexing (once, slow, paid) and querying (every question, fast).
- `Document` is temporary packaging. Put `paper_id` in its metadata so chunks can link back to their paper.
- Use the recursive splitter for messy PDF text. Chunk size decides what retrieval can find.
- Stable ids (`paper_id-cN`) make re-ingestion safe and give citations something to point at.
- Build the store once, load it many times, and use the same embedding model for both.
- Wrap steps in functions and run them from `main()`, so importing the file does nothing expensive.

## Common mistakes
- **Wrong `parents` count.** `Path(__file__).parent.parent / "data"` resolved to `src/data`, a folder that doesn't exist.
- **PDFs in `data/` and not `data/raw/`.** Only `data/raw/*` is git-ignored, so the PDFs would have been committed.
- **Passing a folder to a loader that wants files.** `DoclingLoader(data_dir=...)` isn't a real argument. It takes `file_path`, a file path or a list of paths.
- **Adding 1 twice in the counter.** `index = counts.get(paper_id, 0) + 1` plus `counts[paper_id] = index + 1` gave ids `c1, c3, c5…`.
- **Hard-coding `k`.** `search_kwargs={"k": 3}` ignored the function's `k` argument.
- **Calling `from_documents` without `ids`.** Every run duplicates every chunk.
- **Storing Chroma in a folder git doesn't ignore.** An early version wrote to `data/chroma/`.
- **Creating the LLM in `ingest.py`.** It's unused during indexing, and `ChatOpenAI` needs an API key just to be constructed.
- **Accidental auto-imports.** `from email.mime import text` appeared from an editor auto-import. Remove unused imports (`uv run ruff check` finds them).
- **Using `CharacterTextSplitter` on PDF text.** With no blank lines, it returns one huge chunk.

## Practice
1. Write `tests/test_ingest.py`: give `split_docs` two short fake `Document`s with the same `paper_id`, and assert that the `chunk_index` values are `[0, 1, …]` with no gaps and every `chunk_id` is unique.
2. Write a test that `load_pdfs(tmp_path)` returns `[]` for an empty folder (pytest's `tmp_path` fixture gives you a temporary folder).
3. Change `chunk_size` to 500 and then 2000, delete `.chroma/`, re-ingest, and compare what `check_retrieval()` returns for the same question.
4. Add a function that turns each chunk into your `Chunk` schema model (`id`, `paper_id`, `text`, `index`), and validate all 348.
5. Ask `check_retrieval("What was the utility r for MATH?")` and look at whether the returned chunk still contains the table's header row.

## Teaching tip
Run both splitters on the same real PDF page in front of learners: `CharacterTextSplitter` returns one 13,000-character chunk and `RecursiveCharacterTextSplitter` returns 16 tidy ones, with identical settings. Then ask a question whose answer sits in a table and show the retrieved chunk. Seeing a row of numbers with no header makes the main idea clear: retrieval quality is decided at ingestion time, long before any prompt is written.
