"""
FastAPI backend for the university Q&A chatbot.
Wraps rag_utils.py (same RAG pipeline used by the Streamlit version) behind
a simple REST API so a custom HTML/JS frontend can talk to it.

Run with:
    export OPENAI_API_KEY="your-key-here"
    python build_index.py        # once, or whenever data changes
    uvicorn backend:app --reload --port 8000

Then open static/index.html in a browser (or serve it, see README).
"""

import json
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_utils import get_client, load_index, answer_question

app = FastAPI(title="University Q&A Chatbot API")

# Allow the frontend (served from anywhere, e.g. file:// or a dev server) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_client = None
_index = None
_chunks = None

LOG_PATH = "logs/questions.jsonl"


def log_question(question: str, answer: str, sources: list[str]):
    """Append one question + answer as a JSON line to logs/questions.jsonl."""
    os.makedirs("logs", exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "answered": len(sources) > 0,  # False = the guardrail rejected it (not in data)
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.on_event("startup")
def load_resources():
    global _client, _index, _chunks
    _client = get_client()
    _index, _chunks = load_index()
    print("Loaded index and OpenAI client. Ready.")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer, sources = answer_question(req.question, _client, _index, _chunks, k=3)
    log_question(req.question, answer, sources)
    return AskResponse(answer=answer, sources=sources)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the frontend directly from FastAPI at http://localhost:8000/
app.mount("/", StaticFiles(directory="static", html=True), name="static")
