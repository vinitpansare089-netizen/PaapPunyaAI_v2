
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.services.krishna_service import get_krishna_response


# Initialize FastAPI application
app = FastAPI(
    title="KrishnaAI API",
    description="A Krishna-inspired guidance system powered by RAG.",
    version="1.0.0"
)


# Define request structure
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000
    )


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "KrishnaAI API is running"
    }


# Health-check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# KrishnaAI endpoint
@app.post("/ask-krishna")
def ask_krishna(request: QuestionRequest):

    result = get_krishna_response(request.question)

    return result
