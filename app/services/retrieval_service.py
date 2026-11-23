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


def retrieve_krishna_knowledge(question: str, k: int = 2):

    # Load the existing vector store
    embedding_model = get_embedding_model()

    project_root = Path(__file__).resolve().parents[2]
    save_path = project_root / "vector_store" / "krishna_faiss"

    vector_store = FAISS.load_local(
        str(save_path),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # Retrieve the most relevant chunks
    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results