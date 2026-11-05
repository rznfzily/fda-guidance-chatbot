"""
FastAPI REST API for FDA Guidance Chatbot

Endpoints:
- POST /api/chat - Stream chat responses
- POST /api/query - Non-streaming chat responses
- GET /api/health - Health check
- POST /api/followup - Generate follow-up questions
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import logging
from dotenv import load_dotenv
from langchain_core.documents import Document

from rag.answer import (
    stream_answer,
    answer_question,
    generate_followup_questions
)

load_dotenv(override=True)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FDA Guidance Chatbot",
    description="Chatbot for FDA Guidance Documents",
    version="1.0.0"
)


# Request/Response Models
class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question")
    history: List[Message] = Field(default=[], description="Conversation history")


class SourceDocument(BaseModel):
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="FDA document URL")
    metadata: Dict = Field(default={}, description="Document metadata")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM's answer")
    sources: List[SourceDocument] = Field(..., description="Source documents")
    confidence: float = Field(..., description="Confidence score (0-100)")
    filters: Optional[str] = Field(default=None, description="Filters inferred from the question and applied, e.g. 'CDER, 2023'")


class FollowupRequest(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Answer provided")
    sources: List[SourceDocument] = Field(..., description="Source documents")


class FollowupResponse(BaseModel):
    questions: List[str] = Field(..., description="Follow-up questions")


# Endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FDA Guidance Chatbot"
    }


@app.post("/api/query", response_model=ChatResponse)
def query(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        # Convert Message objects to dict format
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        # Get answer
        answer, docs, confidence, filters = answer_question(
            question=request.question,
            history=history
        )

        # Convert docs to response format
        sources = [
            SourceDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                metadata=doc.metadata
            )
            for doc in docs
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            filters=filters
        )

    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed")


@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events (SSE)."""
    # Convert Message objects to dict format
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # Plain generator: Starlette iterates it in a threadpool, so blocking retrieval/LLM calls
    # don't stall the event loop. Retrieval runs lazily here, so errors must be caught here.
    def event_generator():
        docs_sent = False
        confidence_sent = False
        filters_sent = False

        try:
            for token, docs, confidence, filters in stream_answer(
                question=request.question,
                history=history
            ):
                # Tell the client which inferred filters were applied (once, before the first token)
                if not filters_sent:
                    filters_sent = True
                    if filters:
                        yield f"data: {json.dumps({'type': 'filters', 'description': filters})}\n\n"

                # Send token
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                # Send sources once
                if not docs_sent and docs:
                    sources = [
                        {
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "metadata": doc.metadata
                        }
                        for doc in docs
                    ]
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                    docs_sent = True

                # Send confidence once
                if not confidence_sent and confidence is not None:
                    yield f"data: {json.dumps({'type': 'confidence', 'score': confidence})}\n\n"
                    confidence_sent = True

            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception:
            logger.exception("Streaming failed")
            yield f"data: {json.dumps({'type': 'error', 'message': 'The request failed. Please try again.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/followup", response_model=FollowupResponse)
def generate_followups(request: FollowupRequest):
    """Generate follow-up questions based on Q&A exchange"""
    try:
        docs = [
            Document(
                page_content=src.content,
                metadata=src.metadata
            )
            for src in request.sources
        ]

        questions = generate_followup_questions(
            question=request.question,
            answer=request.answer,
            docs=docs
        )

        return FollowupResponse(questions=questions)

    except Exception:
        logger.exception("Follow-up generation failed")
        raise HTTPException(status_code=500, detail="Follow-up generation failed")


# Run with: uvicorn fda_api_main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
