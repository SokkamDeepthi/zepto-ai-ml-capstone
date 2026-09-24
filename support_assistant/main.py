from fastapi import FastAPI
from pydantic import BaseModel, Field

from support_assistant.graph import graph


app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline deterministic Zepto policy support assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant API is running."
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke({
        "query": request.query
    })

    response = result["response"]

    return AskResponse(
        answer=response["answer"],
        sources=response["sources"],
        confidence=response["confidence"],
    )