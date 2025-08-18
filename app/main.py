# app/main.py
from fastapi import FastAPI
from app.api import rag

app = FastAPI(title="RAG USSD Backend")

app.include_router(rag.router, prefix="/rag", tags=["RAG"])
