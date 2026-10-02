# Zepto Support Assistant

An offline AI-powered support assistant that answers customer questions using Zepto policy documents.

## Overview

This module implements a policy-based question answering system using:

- Sentence Transformers for text embeddings
- ChromaDB for vector storage and similarity search
- LangGraph for the assistant workflow
- FastAPI for the REST API
- Pydantic for request and response validation

The assistant retrieves relevant policy information from local documents and provides an answer along with source documents and a confidence score.

## Project Structure

```text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
├── main.py
└── README.md