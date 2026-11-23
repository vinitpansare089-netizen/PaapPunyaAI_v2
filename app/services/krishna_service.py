from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from app.services.retrieval_service import retrieve_krishna_knowledge

load_dotenv()

def get_krishna_response(question: str):

    ###Retrieve relevant knowledge
    documents = retrieve_krishna_knowledge(question)
