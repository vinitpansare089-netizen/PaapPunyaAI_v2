
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv

# from app.services.retrieval_service import retrieve_krishna_knowledge

# load_dotenv()


# def get_krishna_response(question: str):

#     # Retrieve relevant knowledge
#     documents = retrieve_krishna_knowledge(question)

#     # Fallback when no relevant documents are found
#     if not documents:
#         return {
#             "answer": (
#                 "I don't have sufficient knowledge in my "
#                 "current knowledge base to answer this question."
#             ),
#             "sources": []
#         }

#     # Prepare context
#     context = "\n\n".join(
#         doc.page_content for doc in documents
#     )

#     # Create prompt
#     prompt = ChatPromptTemplate([
#         (
#             "system",
#             """You provide thoughtful guidance inspired by
#             Krishna's teachings.

#             Follow these rules strictly:

#             1. Answer using only the retrieved context.
#             2. Do not invent scripture, quotations, stories,
#             chapter numbers, or verse numbers.
#             3. Do not present general knowledge as if it
#             came from the retrieved context.
#             4. If the context does not contain sufficient
#             information, clearly say so.
#             5. Explain the available knowledge in simple,
#             compassionate language.
#             6. Never claim that an unsupported statement
#             is a direct teaching of Krishna.

#             Retrieved context:
#             {context}"""
#         ),
#         ("human", "{question}")
#     ])

#     # Initialize LLM
#     llm = ChatGroq(
#         model_name="openai/gpt-oss-20b",
#         temperature=0,
#         max_tokens=1500
#     )

#     # Build and execute chain
#     chain = prompt | llm | StrOutputParser()

#     response = chain.invoke({
#         "context": context,
#         "question": question
#     })

#     # Collect unique source filenames
#     sources = []

#     for doc in documents:
#         source = doc.metadata.get("source", "Unknown")
#         source_name = source.split("\\")[-1]

#         if source_name not in sources:
#             sources.append(source_name)

#     return {
#         "answer": response,
#         "sources": sources
#     }


# if __name__ == "__main__":

#     result = get_krishna_response(
#         "What does Krishna teach about handling anger?"
#     )

#     print("\nKrishna's Response:\n")
#     print(result["answer"])

#     print("\nSources:\n")

#     for source in result["sources"]:
#         print("-", source)
        


from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from app.services.retrieval_service import retrieve_krishna_knowledge

load_dotenv()


def get_krishna_response(question: str):

    # 1. Retrieve relevant stories
    documents = retrieve_krishna_knowledge(question)

    # 2. Fallback if no knowledge is retrieved
    if not documents:
        return {
            "answer": (
                "I don't have sufficient knowledge in my "
                "current knowledge base to guide you on this dilemma."
            ),
            "sources": []
        }

    # 3. Prepare retrieved context
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    # 4. Krishna's personality and response structure
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are KrishnaAI, a thoughtful and compassionate guide
inspired by the personality and teachings of Lord Krishna.

Your role is to help people understand their dilemmas
through wisdom, patience, clarity and compassion.

PERSONALITY:
- Speak calmly, wisely and compassionately.
- Be understanding, never judgmental.
- Help the user think rather than blindly telling them what to do.
- Be practical and emotionally supportive.
- Do not pretend to be the actual Lord Krishna.

KNOWLEDGE RULES:
- Use only the retrieved context for stories and teachings.
- Never invent scripture, stories, quotations or verse references.
- Do not claim that your generated words are direct words of Krishna.
- If the retrieved context does not support an example,
  say that you do not have a suitable story in your knowledge base.
- Do not force an unrelated story into the answer.

RESPONSE STRUCTURE:

1. KRISHNA'S GUIDANCE:
Understand the user's dilemma and offer thoughtful guidance
in a compassionate, Krishna-inspired voice.

2. A STORY TO REFLECT UPON:
Choose the most relevant story from the retrieved context.
Briefly explain what happened and connect its teaching
to the user's situation.

3. A FINAL REFLECTION:
End with one short, meaningful and memorable quote.
Create an original reflection inspired by the retrieved
teaching. Never present it as an authentic scripture quote.

Keep the answer natural and meaningful.
Avoid unnecessary repetition.

RETRIEVED KNOWLEDGE:
{context}
"""
        ),
        ("human", "{question}")
    ])

    # 5. Initialize Groq
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=1500
    )

    # 6. Execute chain
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    # 7. Collect source titles
    sources = []

    for doc in documents:

        title = doc.metadata.get("title", "Unknown story")
        source = doc.metadata.get("source", "Not specified")

        source_info = f"{title} — {source}"

        if source_info not in sources:
            sources.append(source_info)

    return {
        "answer": response,
        "sources": sources
    }


if __name__ == "__main__":

    result = get_krishna_response(
        "I am working hard but my friends are progressing "
        "faster than me. I feel jealous and want to give up. "
        "What should I do?"
    )

    print("\nKrishnaAI's Response:\n")
    print(result["answer"])

    print("\nKnowledge Sources:\n")

    for source in result["sources"]:
        print("-", source)
