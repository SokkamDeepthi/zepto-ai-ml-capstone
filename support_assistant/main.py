import json
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.utils import embedding_functions

from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policy_corpus"

# Chroma's default embedding model is all-MiniLM-L6-v2.
# It uses Chroma's ONNX-based implementation instead of
# loading SentenceTransformer/PyTorch directly.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
    description="Offline policy retrieval assistant for the AI/ML capstone.",
)


# ============================================================
# 3. PYDANTIC SCHEMAS
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


# ============================================================
# 4. LANGGRAPH STATE
# ============================================================

class AssistantState(TypedDict, total=False):
    query: str
    intent: str

    retrieved_chunks: list[str]
    retrieved_ids: list[str]
    retrieved_sources: list[str]
    retrieved_distances: list[float]

    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# 5. CHROMADB SETUP
# ============================================================

print("=" * 60)
print("STARTING CHROMADB")
print("=" * 60)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# Use Chroma's built-in ONNX embedding function.
# This avoids importing SentenceTransformer/PyTorch.

embedding_function = embedding_functions.DefaultEmbeddingFunction()


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Zepto policy corpus",
        "hnsw:space": "cosine",
    },
    embedding_function=embedding_function,
)


print("Embedding model:", EMBEDDING_MODEL_NAME)
print("Collection:", COLLECTION_NAME)
print("Storage:", CHROMA_DIR)


# ============================================================
# 6. LOAD EXACTLY 8 POLICY DOCUMENTS
# ============================================================

def load_documents():

    files = sorted(
        DOCS_DIR.glob("doc_*.txt")
    )

    if len(files) != 8:
        raise RuntimeError(
            f"Expected exactly 8 policy documents, "
            f"found {len(files)} in {DOCS_DIR}"
        )

    documents = []
    document_ids = []
    metadatas = []

    for file_path in files:

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            raise RuntimeError(
                f"Empty document found: {file_path.name}"
            )

        documents.append(text)

        document_ids.append(
            file_path.stem
        )

        metadatas.append(
            {
                "source": file_path.name
            }
        )

    return (
        documents,
        document_ids,
        metadatas,
    )


# ============================================================
# 7. BUILD CHROMADB INDEX
# ============================================================

def build_index():

    documents, document_ids, metadatas = load_documents()

    print("=" * 60)
    print("BUILDING CHROMADB INDEX")
    print("=" * 60)

    # Remove old records
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    # Chroma automatically creates embeddings using
    # its built-in embedding function.
    collection.add(
        ids=document_ids,
        documents=documents,
        metadatas=metadatas,
    )

    print("=" * 60)
    print("CHROMADB INDEX READY")
    print("=" * 60)

    print("Documents:", len(documents))
    print("Embedding model:", EMBEDDING_MODEL_NAME)
    print("Collection:", COLLECTION_NAME)
    print("Storage:", CHROMA_DIR)


# ============================================================
# 8. STRUCTURED PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the Zepto policy information provided in the retrieved context.

TASK:
Answer the customer's question using only the retrieved policy context.

FORMAT:
Return valid JSON with exactly these fields:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}

LENGTH:
Keep the answer concise, direct, and relevant.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent, guess, or assume policy details.

CUSTOMER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# ============================================================
# 9. POLICY KEYWORDS
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
    "replacement",
    "payment",
    "order",
    "complaint",
    "damaged",
    "spoiled",
    "incorrect",
]


# ============================================================
# 10. CLASSIFY INTENT
# ============================================================

def classify_intent(
    state: AssistantState
):

    query = state["query"].lower()

    if MOCK_LLM:

        is_policy = any(
            keyword in query
            for keyword in POLICY_KEYWORDS
        )

        intent = (
            "policy_question"
            if is_policy
            else "general_question"
        )

    else:

        intent = classify_with_real_llm(
            query
        )

    print(
        f"[classify_intent] {query} -> {intent}"
    )

    return {
        "intent": intent
    }


# ============================================================
# 11. OPTIONAL REAL LLM INTENT CLASSIFICATION
# ============================================================

def classify_with_real_llm(
    query: str
) -> str:

    try:

        import requests

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            return "general_question"

        prompt = f"""
Classify this query as exactly one of:

policy_question
general_question

Query:
{query}

Return only the classification name.
"""

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": 0,
            },
            timeout=30,
        )

        response.raise_for_status()

        content = (
            response.json()
            ["choices"][0]
            ["message"]
            ["content"]
            .strip()
            .lower()
        )

        if content == "policy_question":
            return "policy_question"

        return "general_question"

    except Exception:
        return "general_question"


# ============================================================
# 12. RETRIEVAL
# ============================================================

def retrieve_top_k(
    query: str,
    k: int = 3,
):

    result = collection.query(
        query_texts=[query],
        n_results=k,
    )

    documents = result["documents"][0]
    ids = result["ids"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    sources = [
        metadata["source"]
        for metadata in metadatas
    ]

    return (
        documents,
        ids,
        sources,
        distances,
    )


# ============================================================
# 13. RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(
    state: AssistantState
):

    query = state["query"]

    (
        documents,
        ids,
        sources,
        distances,
    ) = retrieve_top_k(
        query,
        k=3,
    )

    if not documents:

        return {
            "answer": (
                "No relevant policy information was found."
            ),
            "sources": [],
            "confidence": 0.0,
            "retrieved_chunks": [],
            "retrieved_ids": [],
            "retrieved_sources": [],
            "retrieved_distances": [],
        }

    # Most relevant document
    top_chunk = documents[0].strip()

    # Chroma cosine distance:
    # similarity = 1 - distance

    similarity = max(
        0.0,
        min(
            1.0,
            1.0 - float(distances[0]),
        ),
    )

    if MOCK_LLM:

        answer = (
            "Based on the retrieved context: "
            + top_chunk
        )

    else:

        answer = generate_with_real_llm(
            query=query,
            context=documents,
            source_ids=ids,
        )

    return {
        "answer": answer,
        "sources": ids,
        "confidence": round(
            similarity,
            4,
        ),
        "retrieved_chunks": documents,
        "retrieved_ids": ids,
        "retrieved_sources": sources,
        "retrieved_distances": distances,
    }


# ============================================================
# 14. DIRECT ANSWER
# ============================================================

def direct_answer(
    state: AssistantState
):

    if MOCK_LLM:

        answer = (
            "I can only answer questions about "
            "Zepto policies right now."
        )

    else:

        answer = generate_direct_with_real_llm(
            state["query"]
        )

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0,
    }


# ============================================================
# 15. ROUTING
# ============================================================

def route_by_intent(
    state: AssistantState
):

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# 16. REAL LLM - RETRIEVED CONTEXT
# ============================================================

def generate_with_real_llm(
    query: str,
    context: list[str],
    source_ids: list[str],
) -> str:

    import requests

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        return (
            "ERROR: GROQ_API_KEY is not configured. "
            "Use MOCK_LLM=1 for the required offline mode."
        )

    structured_prompt = PROMPT_TEMPLATE.format(
        query=query,
        context="\n\n".join(context),
    )

    last_error = None

    for attempt in range(3):

        try:

            correction = ""

            if attempt > 0:

                correction = """
Your previous output failed validation.

Return ONLY valid JSON with:
- answer: string
- sources: list of document IDs
- confidence: number from 0 to 1

Do not include Markdown fences.
"""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                structured_prompt
                                + correction
                            ),
                        }
                    ],
                    "temperature": 0,
                },
                timeout=30,
            )

            response.raise_for_status()

            raw_output = (
                response.json()
                ["choices"][0]
                ["message"]
                ["content"]
                .strip()
            )

            if raw_output.startswith("```"):

                raw_output = (
                    raw_output
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            parsed = json.loads(
                raw_output
            )

            validated = AskResponse.model_validate(
                parsed
            )

            validated.sources = [
                source_id
                for source_id in validated.sources
                if source_id in source_ids
            ]

            return validated.answer

        except (
            json.JSONDecodeError,
            ValidationError,
            requests.RequestException,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            last_error = exc

    return (
        "ERROR: Real LLM response failed schema validation "
        f"after 3 attempts. Details: {last_error}"
    )


# ============================================================
# 17. REAL LLM - DIRECT ANSWER
# ============================================================

def generate_direct_with_real_llm(
    query: str
) -> str:

    import requests

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        return (
            "ERROR: GROQ_API_KEY is not configured. "
            "Use MOCK_LLM=1 for the required offline mode."
        )

    prompt = PROMPT_TEMPLATE.format(
        query=query,
        context=(
            "No policy retrieval was performed because "
            "this was classified as a general question."
        ),
    )

    last_error = None

    for attempt in range(3):

        try:

            correction = ""

            if attempt > 0:

                correction = """
Return ONLY valid JSON with:
answer, sources, confidence.

Do not include Markdown fences.
"""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                prompt
                                + correction
                            ),
                        }
                    ],
                    "temperature": 0,
                },
                timeout=30,
            )

            response.raise_for_status()

            raw_output = (
                response.json()
                ["choices"][0]
                ["message"]
                ["content"]
                .strip()
            )

            if raw_output.startswith("```"):

                raw_output = (
                    raw_output
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            parsed = json.loads(
                raw_output
            )

            validated = AskResponse.model_validate(
                parsed
            )

            return validated.answer

        except (
            json.JSONDecodeError,
            ValidationError,
            requests.RequestException,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            last_error = exc

    return (
        "ERROR: Real LLM response failed schema validation "
        f"after 3 attempts. Details: {last_error}"
    )


# ============================================================
# 18. BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    AssistantState
)

graph_builder.add_node(
    "classify_intent",
    classify_intent,
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

graph_builder.add_node(
    "direct_answer",
    direct_answer,
)

graph_builder.add_edge(
    START,
    "classify_intent",
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer":
            "retrieve_and_answer",

        "direct_answer":
            "direct_answer",
    },
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END,
)

graph_builder.add_edge(
    "direct_answer",
    END,
)

graph = graph_builder.compile()


# ============================================================
# 19. INITIALIZE INDEX
# ============================================================

build_index()


# ============================================================
# 20. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "Zepto Support Assistant is running",

        "mock_llm":
            MOCK_LLM,

        "documents_indexed":
            collection.count(),

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "docs":
            "/docs",
    }


# ============================================================
# 21. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "documents_indexed":
            collection.count(),

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "collection":
            COLLECTION_NAME,

        "mock_llm":
            MOCK_LLM,
    }


# ============================================================
# 22. ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest
):

    result = graph.invoke(
        {
            "query":
                request.query.strip()
        }
    )

    response = AskResponse(
        answer=result["answer"],

        sources=result.get(
            "sources",
            [],
        ),

        confidence=result.get(
            "confidence",
            0.0,
        ),
    )

    return response


# ============================================================
# 23. LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )