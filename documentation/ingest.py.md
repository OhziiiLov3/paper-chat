# Ingestion: Turning PDFs into a Searchable Vector Store

**Folder:** `src/paper_chat/` (`ingest.py`), tested in `tests/` (`test_ingest.py`)

## What you'll learn
- How the indexing half of a RAG app works: load → split → embed → store
- How to load PDFs with `pypdf` and turn them into validated `Paper` and `Chunk` models
- How chunk size, overlap, and the recursive splitter shape what retrieval can find
- How stable chunk ids prevent duplicates and make citations possible
- How to test a whole ingestion pipeline with no API calls and no real data

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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
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

| Folder | Holds | Git |
|---|---|---|
| `data/raw/` | Original PDFs. You never edit these. | Ignored, so client documents stay out of the repo |
| `data/processed/` | Files ingestion **generates** (`papers.json`) | Ignored, because it can be rebuilt from `raw/` |
| `.chroma/` | The vector database | Ignored, because it can be rebuilt from `raw/` |

- **`.resolve()`** makes the path absolute, so it doesn't depend on which folder you run the command from.
- **`EMBEDDING_MODEL`** is defined once because ingest and query **must** use the same embedding model. Vectors from two different models can't be compared.

---

## 3. Load: one `Paper` per PDF, one `Document` per page

Loading now produces two things, at two different levels:

| | `Paper` (your schema) | `Document` (LangChain) |
|---|---|---|
| Represents | One whole paper: its facts, not its text | One **page** of text |
| How many (3 project PDFs) | 3 | 92 |
| Lifespan | Permanent, saved to `papers.json` | Temporary, only so the splitter can work |

```python
def build_paper(pdf_path: Path, reader: PdfReader) -> Paper:
    """One Paper per PDF, from the PDF's built-in metadata."""
    meta = reader.metadata
    title = (meta.title if meta else None) or pdf_path.stem
    author_text = (meta.author if meta else None) or ""
    authors = [name.strip() for name in author_text.split(";") if name.strip()]
    return Paper(id=pdf_path.stem, title=title, source=pdf_path.name, authors=authors)


def load_pdfs(raw_dir: Path = RAW_DIR) -> tuple[list[Paper], list[Document]]:
    """Return one Paper per PDF and one Document per page."""
    papers, docs = [], []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        paper = build_paper(pdf_path, reader)
        papers.append(paper)
        for page_number, page in enumerate(reader.pages):
            docs.append(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"paper_id": paper.id, "page": page_number},
                )
            )
    return papers, docs
```

Real output for one of the project papers:

```
{'id': '2604.17031v3', 'title': 'Where is the Mind? Persona Vectors and LLM Individuation',
 'abstract': None, 'source': '2604.17031v3.pdf',
 'authors': ['Pierre Beckmann', 'Patrick Butlin'], 'year': None}
```

First page's metadata: `{'paper_id': '2505.17120v2', 'page': 0}`

| Detail | Why |
|---|---|
| `id = pdf_path.stem` | The filename without `.pdf`. It is **stable**: re-ingesting the same file gives the same id. |
| `paper_id = paper.id` | Every page's `paper_id` comes from the same value as `Paper.id`, so the link is guaranteed. |
| `or pdf_path.stem` / `or ""` | Missing metadata falls back to a value; it doesn't crash. |
| `split(";")` | All 3 project PDFs separate authors with semicolons. Other sources may use commas: this is an assumption about **these** PDFs. |
| `year` and `abstract` are `None` | These PDFs have no date in their metadata, and the abstract is only in the page text. The schema's optional fields are what let this run. |
| `extract_text() or ""` | Image-only (scanned) pages return no text. |
| `-> tuple[...]` | The function returns two lists, so callers unpack them: `papers, docs = load_pdfs()`. |

**The abstract is still searchable.** It is on page 0, so it ends up in the first chunks of each paper. Only the separate `Paper.abstract` field is empty. Extracting it from text works on some papers and picks up page footers on others, so it was left out of v1.

**Why not `PyPDFLoader`?** It lives in `langchain-community`, which LangChain has sunset. Its repository was archived in 2026. Building the `Document` yourself is a few lines and has one less dependency to go stale.

**Library choice for client work (licenses checked on PyPI; get legal review for real contracts):**

| Library | License | Notes |
|---|---|---|
| `pypdf` | BSD-3 | Light, plain text only. Tables lose their column structure. |
| Docling (`langchain-docling`) | MIT | Understands tables, headings, and columns. Heavy (~85 packages). Not tested in this project. |
| PyMuPDF | AGPL-3.0 | Good quality, but AGPL can require open-sourcing the client's app. |

---

## 4. Save the papers

Chroma stores **chunks**, so papers need their own place. A JSON file is the simplest option:

```python
def save_papers(papers: list[Paper], out_dir: Path = PROCESSED_DIR) -> Path:
    """Write the papers to JSON so rag.py can look up titles without re-reading PDFs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "papers.json"
    path.write_text(json.dumps([p.model_dump() for p in papers], indent=2))
    return path
```

- `mkdir(parents=True, exist_ok=True)` creates the folder if needed and does nothing if it already exists.
- `model_dump()` turns each `Paper` into a plain dict, which `json.dumps` can write.
- Later, `rag.py` can show "*Where is the Mind?…*, page 3" in place of a raw chunk id.

---

## 5. Split: text into validated `Chunk`s

A **chunk** is a small passage of text. Retrieval searches chunks, not whole papers, because a focused passage matches a question better and fits in the LLM's prompt.

The splitter decides **where to cut**. Your schema decides **what a valid chunk is**:

```
pages (Document) ──> RecursiveCharacterTextSplitter ──> pieces (Document) ──> Chunk (validated)
```

```python
def split_docs(
    docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Chunk]:
    """Split pages into chunks and validate each one against the Chunk schema."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = splitter.split_documents(docs)

    chunks: list[Chunk] = []
    counts: dict[str, int] = {}
    for piece in pieces:
        paper_id = piece.metadata["paper_id"]
        index = counts.get(paper_id, 0)
        counts[paper_id] = index + 1
        chunks.append(
            Chunk(
                id=f"{paper_id}-c{index}",
                paper_id=paper_id,
                text=piece.page_content,
                index=index,
                page=piece.metadata["page"],
            )
        )
    return chunks
```

Real result: **92 pages → 348 chunks**, sizes from 132 to 1000 characters.

| Parameter | Meaning | Trade-off |
|---|---|---|
| `chunk_size=1000` | Maximum characters per chunk | Too small (e.g. 200) gives one or two sentences, with too little context. Too big makes retrieval less precise. |
| `chunk_overlap=150` | Up to this many characters repeated between neighboring chunks | Keeps a sentence from being cut off with no context on either side. |

`chunk_size` and `chunk_overlap` are parameters with defaults, so tests can use small values (500/50) and you can tune them later without editing the function.

The overlap is a **maximum**, not an exact amount. The splitter only overlaps on clean boundaries (spaces, newlines). Between the first two chunks, the real overlap was 80 characters.

**Why the *recursive* splitter?** It tries separators in order (`"\n\n"` → `"\n"` → `" "` → `""`) until every piece fits. The simpler `CharacterTextSplitter` uses only one separator (`"\n\n"`). `pypdf` text has no blank lines, so on the first 3 pages of one paper (real output):

```
CharacterTextSplitter:          1 chunk,   size 13,124   ← ignored chunk_size
RecursiveCharacterTextSplitter: 16 chunks, sizes 154–992
```

Each page is split separately, so **a chunk never spans two pages**. That keeps `page` accurate for citations.

> **Key point:** Chunk size decides what retrieval *can* find. If an answer is split across two chunks, or a table loses its header, no prompt can fix it later.

---

## 6. Stable chunk ids: numbering within each paper

The loop in `split_docs` numbers chunks **within their own paper**:

```python
index = counts.get(paper_id, 0)  # chunks already numbered for this paper
counts[paper_id] = index + 1  # the next one gets the next number
```

- `counts` tracks how many chunks each paper has so far. `.get(paper_id, 0)` returns 0 for a paper's first chunk.
- That count **is** the new chunk's 0-based index. Add 1 in one place only: the update.

Real result: chunks per paper `{'2605.13339v4': 154, '2604.17031v3': 118, '2505.17120v2': 76}`, each numbered 0, 1, 2… with no gaps. All 348 ids are unique. Second chunk:

```
Chunk(id='2505.17120v2-c1', paper_id='2505.17120v2', page=0, text='possible to improve these capabilities...', index=1)
```

Short, readable ids like `2505.17120v2-c1` are what the LLM copies into a `Citation.chunk_id`. Because every chunk is a `Chunk`, a bad value (a missing `paper_id`, an `index` that isn't a number) fails **here, at ingestion time**, before anything is stored.

---

## 7. Embed and store: build once, load many times

An **embedding** is a list of numbers that represents the meaning of a text. Texts with similar meaning get similar numbers, which is why this kind of search is called *semantic*. A **vector store** (here, Chroma) saves those numbers and finds the closest ones to a question.

Chroma works with LangChain `Document`s, so chunks are converted back **only at that boundary**:

```python
def to_documents(chunks: list[Chunk]) -> list[Document]:
    """Convert Chunks back to LangChain Documents, only at the Chroma boundary."""
    return [
        Document(
            page_content=c.text,
            metadata={
                "chunk_id": c.id,
                "paper_id": c.paper_id,
                "chunk_index": c.index,
                "page": c.page,
            },
        )
        for c in chunks
    ]


def build_vectorstore(chunks: list[Chunk], embeddings=None) -> Chroma:
    """Embed chunks and write them to Chroma. Costs API calls with real embeddings."""
    return Chroma.from_documents(
        to_documents(chunks),
        embeddings or get_embeddings(),
        ids=[c.id for c in chunks],
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )


def load_vectorstore() -> Chroma:
    """Load a Chroma vector store from the persisted directory."""
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
```

Real stored metadata for one chunk: `{'chunk_id': '2505.17120v2-c1', 'paper_id': '2505.17120v2', 'chunk_index': 1, 'page': 0}`

| Function | Cost | Used by |
|---|---|---|
| `build_vectorstore` | Paid: embeds every chunk | `main()`, once per ingest |
| `load_vectorstore` | Free: opens what's on disk | `rag.py`, every run |

- **Your models everywhere, LangChain types only at the edges.** Everything past `split_docs` works with `Chunk`. If you swap Chroma for another store, only `to_documents` and `build_vectorstore` change.
- **The metadata keys are a contract with `rag.py`**, which reads `chunk_id` and `page`. Renaming one would break `rag.py` silently. A test checks them (section 9).
- **`embeddings=None`:** callers can pass their own embeddings. Tests pass `FakeEmbeddings` (no API calls); normal runs fall back to `get_embeddings()`.

**Why `ids=` matters.** With ids, `langchain-chroma` calls Chroma's `upsert` (checked in the installed source). Re-running ingestion *replaces* chunks with the same id. Without ids, every run **adds another copy**. Real check with fake embeddings: without `ids`, 3 chunks became **6 stored after 2 runs**.

Upsert won't delete ids that no longer exist, and it doesn't reshape old metadata. If you change chunk size, numbering, or metadata keys, delete `.chroma/` and ingest again.

**Intel Macs:** `chromadb` needs `onnxruntime`, whose newest versions have no Intel Mac builds. The project's `pyproject.toml` has a `[tool.uv] required-environments` setting so uv picks a compatible version there.

---

## 8. Retrieve and sanity-check

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

## 9. Functions plus `main()`: importable, not automatic

```python
def main() -> None:
    papers, docs = load_pdfs()
    print(f"Saved {len(papers)} papers to {save_papers(papers)}")
    chunks = split_docs(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    build_vectorstore(chunks)
    print(f"Stored in {CHROMA_DIR}")
    check_retrieval()


if __name__ == "__main__":
    main()
```

`if __name__ == "__main__":` is true only when you run the file directly (`uv run python -m paper_chat.ingest`), not when another file imports it. So `from paper_chat.ingest import get_retriever` in `rag.py` doesn't re-read PDFs or re-embed anything.

Code at the top level of a module runs on **every** import. Before this file was refactored, importing it would have re-embedded all chunks. Small, separate functions are also what make each step **testable**.

---

## 10. Testing the pipeline with no API calls

`tests/test_ingest.py` checks the pipeline end to end without an API key, without the real PDFs, and without touching your real `.chroma/`. The same pattern as the schema tests applies ([schema.py.md](schema.py.md)): test your decisions, one behavior per test.

**Two helpers make the test data:**

```python
def make_pdf(path: Path, pages: int = 1, metadata: dict | None = None) -> Path:
    """Write a small blank PDF for tests."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    if metadata:
        writer.add_metadata(metadata)
    with open(path, "wb") as f:
        writer.write(f)
    return path


def make_page(paper_id: str, page: int, words: int = 300) -> Document:
    """A fake page long enough to split into several chunks."""
    text = " ".join(f"word{i}" for i in range(words))
    return Document(page_content=text, metadata={"paper_id": paper_id, "page": page})
```

- `make_pdf` uses `pypdf`'s `PdfWriter` to **create** PDFs, so each test controls exactly what's in them. Metadata keys start with `/` (`"/Title"`, `"/Author"`) because that's how the PDF format names them.
- `make_page` skips files entirely. `split_docs` only needs `Document`s, and 300 words is guaranteed to need several chunks.

**The 5 tests, ordered by what would hurt most if it broke:**

| # | Test | Protects | If it broke |
|---|---|---|---|
| 1 | `test_load_pdfs_one_paper_per_pdf_one_doc_per_page` | The page ↔ paper link | Chunks wouldn't know their paper, so citations would be meaningless |
| 2 | `test_build_paper_falls_back_when_metadata_missing` | Handling messy PDFs | One PDF without metadata would crash the whole ingest |
| 3 | `test_split_docs_numbers_chunks_per_paper` | The chunk counter | Gaps or repeated ids. This bug **actually happened** (`c1, c3, c5…`) |
| 4 | `test_split_docs_respects_size_and_keeps_page` | Chunk size and page tracking | Chunks too big to retrieve precisely, or wrong pages in citations |
| 5 | `test_build_vectorstore_stores_once_and_rerun_does_not_duplicate` | Upsert and the metadata contract | Duplicate chunks on every run, or `rag.py` broken by a renamed key |

Test 5 shows the pytest tools that keep tests safe and free:

```python
def test_build_vectorstore_stores_once_and_rerun_does_not_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma")
    chunks = split_docs([make_page("a", 0)], chunk_size=500, chunk_overlap=50)
    fake = FakeEmbeddings(size=8)

    build_vectorstore(chunks, fake)
    store = build_vectorstore(chunks, fake)

    stored = store.get(ids=[chunks[0].id])
    assert store._collection.count() == len(chunks)
    assert stored["metadatas"][0] == {
        "chunk_id": "a-c0",
        "paper_id": "a",
        "chunk_index": 0,
        "page": 0,
    }
```

| Tool | What it does |
|---|---|
| `tmp_path` | A fresh, empty temporary folder for each test, cleaned up afterwards |
| `monkeypatch.setattr(ingest, "CHROMA_DIR", ...)` | Temporarily replaces a module's value for one test, then restores it. Here it points Chroma at the temp folder |
| `FakeEmbeddings(size=8)` | From `langchain_core`: random vectors, no API call. Fine for checking **storage**, not search quality |

Run them with `uv run pytest -v tests/test_ingest.py`.

**Deliberately not unit-tested:** `get_retriever` and `check_retrieval` with real embeddings. Whether the *right* chunks come back is retrieval **quality**, which belongs in an evaluation set (questions paired with the chunk that should be found), not in unit tests.

> **Key point:** Design functions so tests can replace the expensive parts. A `raw_dir` parameter, `chunk_size` parameters, and `embeddings=None` are what make this whole pipeline testable for free.

---

## Key ideas
- RAG is two pipelines: indexing (once, slow, paid) and querying (every question, fast).
- Your schemas are the data model; LangChain's `Document` appears only at the splitter and Chroma boundaries.
- Use the recursive splitter for messy PDF text. Chunk size decides what retrieval can find.
- Stable ids (`paper_id-cN`) make re-ingestion safe and give citations something to point at.
- Build the store once, load it many times, and use the same embedding model for both.
- Parameters with defaults (`raw_dir`, `chunk_size`, `embeddings`) make every step testable without API calls.

## Common mistakes
- **Wrong `parents` count.** `Path(__file__).parent.parent / "data"` resolved to `src/data`, a folder that doesn't exist.
- **PDFs in `data/` and not `data/raw/`.** Only `data/raw/*` is git-ignored, so the PDFs would have been committed.
- **Forgetting to git-ignore generated output.** `data/processed/papers.json` would have been committed without its `.gitignore` entry.
- **Passing a folder to a loader that wants files.** `DoclingLoader(data_dir=...)` isn't a real argument. It takes `file_path`, a file path or a list of paths.
- **Adding 1 twice in the counter.** `index = counts.get(paper_id, 0) + 1` plus `counts[paper_id] = index + 1` gave ids `c1, c3, c5…`.
- **Building a `Chunk` with a field the schema doesn't define.** Pydantic silently drops it. `page` had to be added to `Chunk` first.
- **Hard-coding `k`.** `search_kwargs={"k": 3}` ignored the function's `k` argument.
- **Calling `from_documents` without `ids`.** Every run duplicates every chunk.
- **Storing Chroma in a folder git doesn't ignore.** An early version wrote to `data/chroma/`.
- **Creating the LLM in `ingest.py`.** It's unused during indexing, and `ChatOpenAI` needs an API key just to be constructed.
- **Accidental auto-imports.** `from email.mime import text` appeared from an editor auto-import. `uv run ruff check` finds unused imports and sorts the rest.
- **Using `CharacterTextSplitter` on PDF text.** With no blank lines, it returns one huge chunk.
- **Tests that write to the real store.** Without `monkeypatch` and `tmp_path`, a test run would overwrite `.chroma/`.

## Practice
1. Add a 6th test: `build_paper` on a PDF with `"/Author": "Ada Lovelace; Alan Turing"` returns `["Ada Lovelace", "Alan Turing"]`. Then change `split(";")` to `split(",")` and watch only that test fail.
2. Write a test for `save_papers(papers, tmp_path)`: read the JSON back, rebuild each entry with `Paper(**entry)`, and check it equals the original.
3. Write a test that `load_pdfs(tmp_path)` returns `([], [])` for an empty folder.
4. Change `chunk_size` to 500 and then 2000, delete `.chroma/`, re-ingest, and compare what `check_retrieval()` returns for the same question.
5. Ask `check_retrieval("What was the utility r for MATH?")` and check whether the returned chunk still contains the table's header row.

## Teaching tip
Run both splitters on the same real PDF page in front of learners: `CharacterTextSplitter` returns one 13,000-character chunk and `RecursiveCharacterTextSplitter` returns 16 tidy ones, with identical settings. Then ask a question whose answer sits in a table and show the retrieved chunk. Seeing a row of numbers with no header makes the main idea clear: retrieval quality is decided at ingestion time, long before any prompt is written. For testing, remove `ids=` from `build_vectorstore`, run test 5, and show it fail with 6 stored for 3 chunks: learners see why "re-running is safe" is a property you prove, not assume.
