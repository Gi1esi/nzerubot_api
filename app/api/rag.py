# app/api/rag.py
from fastapi import APIRouter
from app.models.rag_models import QuestionRequest, AnswerResponse
from app.core.pipeline import answer_question

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)
    return AnswerResponse(answer=answer)
