"""
Ingestion services 
"""


from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_krishna_documents():

    # Locate the Krishna knowledge directory
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "app" / "data" / "krishna"

    # Load all text files
    loader = DirectoryLoader(
        str(data_path),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

    documents = loader.load()

    return documents


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks