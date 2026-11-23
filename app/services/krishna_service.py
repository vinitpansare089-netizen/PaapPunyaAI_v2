from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from app.services.retrieval_service import retrieve_krishna_knowledge

load_dotenv()

def get_krishna_response(question: str):

    ###Retrieve relevant knowledge
    documents = retrieve_krishna_knowledge(question)

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    ###Creating the Prompt 
    prompt = ChatPromptTemplate([
        (
            "system",
            """You are Krishna, offering thoughtful guidance inspired
            by the provided knowledge.

            Use the retrieved context to guide your response.
            Do not invent scripture, quotations, or events.
            If the context is insufficient, acknowledge the limitation.
            Be compassionate, clear, and practical.

            Retrieved knowledge:
            {context}"""
        ),

        ('human', '{question}')
    ])

    ###Initialize the LLM
    llm = ChatGroq(
        model_name = 'openai/gpt-oss-20b',
        temperature = 0,
        max_tokens = 1500
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        'context' : context,
        'question': question
    })

    return response 