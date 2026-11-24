from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings



DATA_DIR = Path(__file__).parent.parent / "data"

help(DATA_DIR)