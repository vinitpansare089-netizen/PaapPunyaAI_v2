
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from app.services.retrieval_service import retrieve_krishna_knowledge

load_dotenv()


def get_krishna_response(question: str):

    # Retrieve relevant knowledge
    documents = retrieve_krishna_knowledge(question)

    # Fallback when no relevant documents are found
    if not documents:
        return {
            "answer": (
                "I don't have sufficient knowledge in my "
                "current knowledge base to answer this question."
            ),
            "sources": []
        }

    # Prepare context
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    # Create prompt
    prompt = ChatPromptTemplate([
        (
            "system",
            """You provide thoughtful guidance inspired by
            Krishna's teachings.

            Follow these rules strictly:

            1. Answer using only the retrieved context.
            2. Do not invent scripture, quotations, stories,
            chapter numbers, or verse numbers.
            3. Do not present general knowledge as if it
            came from the retrieved context.
            4. If the context does not contain sufficient
            information, clearly say so.
            5. Explain the available knowledge in simple,
            compassionate language.
            6. Never claim that an unsupported statement
            is a direct teaching of Krishna.

            Retrieved context:
            {context}"""
        ),
        ("human", "{question}")
    ])

    # Initialize LLM
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=1500
    )

    # Build and execute chain
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    # Collect unique source filenames
    sources = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        source_name = source.split("\\")[-1]

        if source_name not in sources:
            sources.append(source_name)

    return {
        "answer": response,
        "sources": sources
    }


if __name__ == "__main__":

    result = get_krishna_response(
        "What does Krishna teach about handling anger?"
    )

    print("\nKrishna's Response:\n")
    print(result["answer"])

    print("\nSources:\n")

    for source in result["sources"]:
        
        print("-", source)