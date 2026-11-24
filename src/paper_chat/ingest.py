from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# Path to pdfs in data folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


llm = ChatOpenAI(model_name="gpt-5.4-mini", temperature=0, max_tokens=1000)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Step 1: Load Documents from from data directory
docs = []
for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
    reader = PdfReader(pdf_path)
    for page_number, page in enumerate(reader.pages):
        docs.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={"paper_id": pdf_path.stem, "source": pdf_path.name, "page": page_number},
            )
        )

# Step 2: Split Docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"{len(docs)} documents -> {len(chunks)} chunks")
