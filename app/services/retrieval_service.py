# from pathlib import Path

# from langchain_community.vectorstores import FAISS

# from app.services.ingestion_service import (
#     load_krishna_documents,
#     split_documents
# )

# from app.services.embedding_service import get_embedding_model

# def create_krishna_vector_store():

#     ###Loading the documents
#     documents = load_krishna_documents()

#     ### splits documents into chunks
#     chunks = split_documents(documents)

#     ### getting the embedding model
#     embeddings = get_embedding_model()

#     ### Convet chunks into vector and create FAISS store
#     vector_store = FAISS.from_documents(
#         documents = chunks,
#         embedding = embeddings
#     )


#     project_root = Path(__file__).resolve().parents[2]
#     save_path = project_root / "vector_store" / "krishna_faiss"

#     vector_store.save_local(str(save_path))

#     return vector_store


# def retrieve_krishna_knowledge(question: str, k: int = 2):

#     # Load the existing vector store
#     embedding_model = get_embedding_model()

#     project_root = Path(__file__).resolve().parents[2]
#     save_path = project_root / "vector_store" / "krishna_faiss"

#     vector_store = FAISS.load_local(
#         str(save_path),
#         embedding_model,
#         allow_dangerous_deserialization=True
#     )

#     # Retrieve the most relevant chunks
#     results = vector_store.similarity_search_with_score(
#     question,
#     k=k
# )

#     filtered_results = [
#         doc for doc, score in results
#         if score < 1.0
#     ]

#     return filtered_results




from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.services.ingestion_service import load_krishna_documents
from app.services.embedding_service import get_embedding_model


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAISS_PATH = PROJECT_ROOT / "vector_store" / "krishna_faiss"


# Shared model and vector store
_embedding_model = None
_vector_store = None


def get_vector_store():

    global _embedding_model, _vector_store

    # Load embedding model only once
    if _embedding_model is None:
        _embedding_model = get_embedding_model()

    # Load FAISS index only once
    if _vector_store is None:

        if not FAISS_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. "
                "Run create_krishna_vector_store() first."
            )

        _vector_store = FAISS.load_local(
            str(FAISS_PATH),
            _embedding_model,
            allow_dangerous_deserialization=True
        )

    return _vector_store


def create_krishna_vector_store():

    global _vector_store

    # Load JSON documents
    documents = load_krishna_documents()

    # Load embedding model
    embeddings = get_embedding_model()

    # Create FAISS index
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Create directory if necessary
    FAISS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save index
    vector_store.save_local(str(FAISS_PATH))

    # Update shared vector store
    _vector_store = vector_store

    print(
        f"FAISS index created with {len(documents)} documents."
    )

    return vector_store




def retrieve_krishna_knowledge(
    question: str,
    k: int = 2,
    max_distance: float = 1.70
):

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    # Keep only sufficiently relevant documents
    relevant_documents = []

    for doc, score in results:

        print("\nStory:", doc.metadata.get("title"))
        print("FAISS distance:", score)

        if score <= max_distance:
            relevant_documents.append(doc)

    return relevant_documents



if __name__ == "__main__":

    test_questions = [
        "I feel jealous because my friends are more successful than me.",
        "Explain quantum computing and quantum entanglement."
    ]

    # Build the FAISS index once
    create_krishna_vector_store()

    # Test retrieval for each dilemma
    for question in test_questions:

        print("\n" + "=" * 60)
        print(f"USER DILEMMA: {question}")
        print("=" * 60)

        results = retrieve_krishna_knowledge(
            question,
            k=2
        )

        for index, doc in enumerate(results, start=1):

            print(f"\nRetrieved Story {index}:")
            print("Title:", doc.metadata.get("title"))
            print("Story:", doc.page_content)
