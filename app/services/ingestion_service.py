# """
# Ingestion services 
# """


# from pathlib import Path

# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# def load_krishna_documents():

#     # Locate the Krishna knowledge directory
#     project_root = Path(__file__).resolve().parents[2]
#     data_path = project_root / "app" / "data" / "krishna"

#     # Load all text files
#     loader = DirectoryLoader(
#         str(data_path),
#         glob="*.txt",
#         loader_cls=TextLoader,
#         loader_kwargs={"encoding": "utf-8"},
#         show_progress=True
#     )

#     documents = loader.load()

#     return documents


# def split_documents(documents):

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 300,
#         chunk_overlap = 50,
#         separators=["\n\n", "\n", " ", ""]
#     )

#     chunks = text_splitter.split_documents(documents)

#     return chunks



"""
Ingestion service
Loads Krishna stories from JSON
and converts them into LangChain Documents.
"""

import json
from pathlib import Path

from langchain_core.documents import Document


def load_krishna_documents():

    # Locate project directory
    project_root = Path(__file__).resolve().parents[2]

    json_path = (
        project_root
        / "app"
        / "data"
        / "krishna"
        / "Krishna_stories.json"
    )

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    stories = data.get("stories", data.get("documents", []))

    documents = []

    # Convert each story into a LangChain Document
    for story in stories:

        content = f"""
Title: {story['title']}

Story:
{story['story']}

Teaching:
{story['teaching']}

Tags:
{', '.join(story.get('tags', []))}
"""

        metadata = {
            "id": story["id"],
            "deity": story.get("deity", "Krishna"),
            "title": story["title"],
            "source": story.get("source", "Not specified"),
            "document_type": story.get("document_type", "story")
        }

        document = Document(
            page_content=content.strip(),
            metadata=metadata
        )

        documents.append(document)

    return documents


if __name__ == "__main__":

    docs = load_krishna_documents()

    print(f"\nTotal documents loaded: {len(docs)}")

    for doc in docs[:2]:
        print("\n--------------------")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
