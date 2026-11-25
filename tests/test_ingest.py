"""Tests for the ingest module in paper_chat."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from pypdf import PdfReader, PdfWriter

from paper_chat import ingest
from paper_chat.ingest import build_paper, build_vectorstore, load_pdfs, split_docs


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


# 1. Loading: one Paper per PDF, one Document per page, linked by paper_id
def test_load_pdfs_one_paper_per_pdf_one_doc_per_page(tmp_path):
    make_pdf(tmp_path / "a.pdf", pages=2)
    make_pdf(tmp_path / "b.pdf", pages=3)

    papers, docs = load_pdfs(tmp_path)

    assert [p.id for p in papers] == ["a", "b"]
    assert len(docs) == 5
    assert docs[0].metadata == {"paper_id": "a", "page": 0}
    assert docs[2].metadata == {"paper_id": "b", "page": 0}


# 2. Messy PDFs: missing metadata falls back instead of crashing
def test_build_paper_falls_back_when_metadata_missing(tmp_path):
    pdf = make_pdf(tmp_path / "no-meta.pdf")
    paper = build_paper(pdf, PdfReader(pdf))

    assert paper.title == "no-meta"
    assert paper.authors == []


# 3. Chunk numbering: 0, 1, 2... per paper, no gaps, restarts for each paper
def test_split_docs_numbers_chunks_per_paper(tmp_path):
    docs = [make_page("a", 0), make_page("a", 1), make_page("b", 0)]

    chunks = split_docs(docs, chunk_size=500, chunk_overlap=50)

    a_indexes = [c.index for c in chunks if c.paper_id == "a"]
    b_indexes = [c.index for c in chunks if c.paper_id == "b"]
    assert a_indexes == list(range(len(a_indexes)))
    assert b_indexes == list(range(len(b_indexes)))
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].id == "a-c0"


# 4. Chunks respect the size limit and keep their page
def test_split_docs_respects_size_and_keeps_page():
    docs = [make_page("a", 0), make_page("a", 7)]

    chunks = split_docs(docs, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 2
    assert all(len(c.text) <= 500 for c in chunks)
    assert {c.page for c in chunks} == {0, 7}


# 5. Storing: every chunk stored once, and re-running doesn't duplicate (upsert)
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
