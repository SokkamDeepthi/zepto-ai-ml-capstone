import os
from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from support_assistant.knowledge_base import retrieve


# ============================================================
# SETTINGS
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1")


# ============================================================
# PYDANTIC OUTPUT SCHEMA
# ============================================================

class AssistantResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list
    response: dict


# ============================================================
# STRUCTURED PROMPT SKELETON
# ============================================================

PROMPT_SKELETON = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the policy information provided in the retrieved context.

TASK:
Answer the customer's question using the retrieved policy context.

FORMAT:
Give a clear and concise answer.

LENGTH:
Keep the answer short and useful.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.

FEW-SHOT EXAMPLE:

Customer:
How long does Zepto delivery take?

Context:
Zepto delivers grocery and household essentials within 10 to 30 minutes
of order confirmation depending on the delivery zone and current order volume.

Answer:
Based on the retrieved context: Zepto delivers within 10 to 30 minutes
of order confirmation, depending on the delivery zone and current order volume.
"""


# ============================================================
# NODE 1 — CLASSIFY INTENT
# ============================================================

def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    print(f"Intent: {intent}")

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# NODE 2 — RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    results = retrieve(query, top_k=3)

    if not results:
        response = AssistantResponse(
            answer="No relevant policy information was found.",
            sources=[],
            confidence=0.0,
        )

        return {
            **state,
            "retrieved_chunks": [],
            "response": response.model_dump(),
        }

    top_result = results[0]

    snippet = top_result["text"][:200].strip()

    answer = f"Based on the retrieved context: {snippet}"

    sources = [
        result["chunk_id"]
        for result in results
    ]

    # Convert Chroma distance to a simple confidence value.
    distance = top_result["distance"]

    confidence = max(0.0, min(1.0, 1.0 - float(distance)))

    response = AssistantResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
    )

    return {
        **state,
        "retrieved_chunks": results,
        "response": response.model_dump(),
    }


# ============================================================
# NODE 3 — DIRECT ANSWER
# ============================================================

def direct_answer(state: GraphState) -> GraphState:
    response = AssistantResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0,
    )

    return {
        **state,
        "response": response.model_dump(),
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_intent(state: GraphState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(GraphState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("TEST 1 — POLICY QUESTION")
    print("========================================")

    result1 = graph.invoke({
        "query": "How long does Zepto delivery take?"
    })

    print("\nFinal Response:")
    print(result1["response"])


    print("\n========================================")
    print("TEST 2 — GENERAL QUESTION")
    print("========================================")

    result2 = graph.invoke({
        "query": "What is the capital of India?"
    })

    print("\nFinal Response:")
    print(result2["response"])