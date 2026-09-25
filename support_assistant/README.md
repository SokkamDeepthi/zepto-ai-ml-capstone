# Zepto Support Assistant

This module implements an offline Zepto Support Assistant using policy documents, local sentence-transformer embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

## Features

* Loads 8 Zepto policy documents.
* Creates document chunks and local embeddings using `all-MiniLM-L6-v2`.
* Stores embeddings in ChromaDB.
* Uses LangGraph for intent routing and retrieval/answer generation.
* Uses deterministic mock mode by default through `MOCK_LLM`.
* Returns a structured response containing `answer`, `sources`, and `confidence`.
* Provides a FastAPI `/ask` endpoint.
* Includes a Dockerfile for local container execution.

## Documents

The policy corpus contains:

* `doc_01.txt` — Delivery Policy
* `doc_02.txt` — Returns & Refunds
* `doc_03.txt` — Membership Tiers
* `doc_04.txt` — Order Tracking
* `doc_05.txt` — Order Cancellation Policy
* `doc_06.txt` — Damaged or Missing Items
* `doc_07.txt` — Gift Cards
* `doc_08.txt` — Customer Support Hours

## Architecture

The complete RAG pipeline follows:

```text
Policy Documents
      ↓
Ingestion / Chunking
      ↓
Embedding with all-MiniLM-L6-v2
      ↓
ChromaDB Vector Collection
      ↓
User Query
      ↓
LangGraph classify_intent
      ↓
 ┌───────────────────────┐
 │                       │
policy_question     general_question
 │                       │
 ↓                       ↓
retrieve_and_answer   direct_answer
 │                       │
 ↓                       ↓
Final Pydantic Response
```

### 1. Ingestion

The eight policy documents are stored in the `support_assistant` directory. The knowledge-base implementation reads the documents and creates a chunk for each document. Each chunk is assigned a document/chunk identifier such as `doc_01_chunk_01`.

### 2. Embedding

The chunks are embedded locally using the `all-MiniLM-L6-v2` sentence-transformer model. No paid API or LLM provider is required for embeddings.

The generated vectors are stored in the ChromaDB collection under:

```text
support_assistant/chroma_db/
```

### 3. Retrieval

For policy questions, the LangGraph `retrieve_and_answer` node embeds the incoming query and retrieves the top-3 most similar chunks from ChromaDB using cosine similarity.

The retrieved chunk IDs are returned in the `sources` field of the final response.

### 4. Generation

In the default mock mode, the final answer does not call an external LLM. The response is generated deterministically from the most similar retrieved chunk using the template:

```text
Based on the retrieved context: {top_chunk_snippet}
```

For general questions, the `direct_answer` node returns a fixed response without retrieval.

### MOCK_LLM behavior

`MOCK_LLM` controls only the LLM-generation/classification branches.

With `MOCK_LLM` unset or set to `1`:

* intent classification uses the required keyword heuristic;
* policy questions use real ChromaDB retrieval;
* answer generation uses deterministic mock logic;
* general questions use the fixed direct-answer response;
* no external LLM API call is required.

With `MOCK_LLM=0`, the optional real-LLM path can be used instead. Retrieval still happens through the local embedding model and ChromaDB.

## Structured Prompt

The optional real-LLM prompt follows the role-context-task-format-length structure and contains a negative grounding constraint and a few-shot example.

The grounding constraint requires the model not to use information that is absent from the retrieved context.

A few-shot example is included to demonstrate the expected grounded answer format.

## LangGraph Workflow

The graph contains three main nodes:

1. `classify_intent`
2. `retrieve_and_answer`
3. `direct_answer`

The `classify_intent` node routes the query conditionally:

```text
policy_question → retrieve_and_answer
general_question → direct_answer
```

The routing itself does not require an LLM.

## Structured Response

The final API response follows the Pydantic schema:

```json
{
  "answer": "string",
  "sources": ["document/chunk IDs"],
  "confidence": 0.0
}
```

For policy questions, `sources` contains the retrieved chunk IDs.

For general questions, `sources` is empty.

## Running the FastAPI Service

From the project root:

```powershell
uvicorn support_assistant.main:app --port 8001
```

The API is available at:

```text
http://127.0.0.1:8001
```

Swagger documentation:

```text
http://127.0.0.1:8001/docs
```

## Example API Calls

The following examples were tested with `MOCK_LLM` left at its default state.

### Example 1 — Policy Question

Request:

```json
{
  "query": "What is the delivery policy?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of \norder confirmation, depending on the customer's delivery zone and current order volume. Standard \nd",
  "sources": [
    "doc_01_chunk_01",
    "doc_02_chunk_01",
    "doc_05_chunk_01"
  ],
  "confidence": 0.42414945363998413
}
```

This query triggered the policy-question route and retrieved `doc_01_chunk_01`, which contains the delivery policy.

### Example 2 — General Question

Request:

```json
{
  "query": "What is the capital of France?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This query triggered the general-question route and therefore did not perform policy retrieval.

## Docker

The FastAPI service includes a `Dockerfile` for local container execution.

Build the image:

```powershell
docker build -t zepto-support-assistant ./support_assistant
```

Run the container:

```powershell
docker run -p 7860:7860 zepto-support-assistant
```

The API can then be accessed through:

```text
http://127.0.0.1:7860/ask
```

## Files

```text
support_assistant/
├── Dockerfile
├── README.md
├── doc_01.txt
├── doc_02.txt
├── doc_03.txt
├── doc_04.txt
├── doc_05.txt
├── doc_06.txt
├── doc_07.txt
├── doc_08.txt
├── graph.py
├── knowledge_base.py
├── main.py
└── chroma_db/
```

## Purpose

The purpose of this module is to demonstrate a grounded support assistant that retrieves information from an internal policy corpus rather than relying only on general model knowledge. The graded baseline is fully offline and deterministic when `MOCK_LLM` is left at its default state.
