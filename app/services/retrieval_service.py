from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.services.ingestion_service import (
    load_krishna_documents,
    split_documents
)

from app.services.embedding_service import get_embedding_model

def create_krishna_vector_store():

    ###Loading the documents
    documents = load_krishna_documents()

    ### splits documents into chunks
    chunks = split_documents(documents)

    ### getting the embedding model
    embeddings = get_embedding_model()

    ### Convet chunks into vector and create FAISS store
    vector_store = FAISS.from_documents(
        documents = chunks,
        embedding = embeddings
    )


    project_root = Path(__file__).resolve().parents[2]
    save_path = project_root / "vector_store" / "krishna_faiss"

    vector_store.save_local(str(save_path))

    return vector_store