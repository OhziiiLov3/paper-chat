"""Indexing pipeline: load PDFs -> split into chunks -> embed -> store in Chroma."""

import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from paper_chat.schema import Chunk, Paper

load_dotenv()


# Path to pdfs in data folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "llm_academic_papers"
EMBEDDING_MODEL = "text-embedding-3-small"


# Step 1: Build Paper + Load Documents from raw data directory


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


def save_papers(papers: list[Paper], out_dir: Path = PROCESSED_DIR) -> Path:
    """Write the papers to JSON so rag.py can look up titles without re-reading PDFs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "papers.json"
    path.write_text(json.dumps([p.model_dump() for p in papers], indent=2))
    return path


# Step 2: Split Docs into chunks
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


# Step 3: Create a Chroma vector store and add the chunks
def get_embeddings() -> OpenAIEmbeddings:
    """Get embeddings from OpenAI."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


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


# Step 4: Retrieve, turn vector store into a retriever
def get_retriever(k: int = 3):
    """Get a retriever from the Chroma vector store."""
    return load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": k})


# Step 5: Sanity check, see what the retriever returns for a sample question
def check_retrieval(query: str = "What is the main contribution of the paper?", k: int = 3) -> None:
    """Print the top-k chunks for a query so you can eyeball retrieval quality."""
    retriever = get_retriever(k=k)
    print(f"\nQuery: {query}")
    for doc in retriever.invoke(query):
        print(f"- {doc.metadata['chunk_id']} (page {doc.metadata['page']})")
        print(f"  {doc.page_content[:200]!r}")


# === Run the indexing pipeline: `uv run python -m paper_chat.ingest` ===
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
