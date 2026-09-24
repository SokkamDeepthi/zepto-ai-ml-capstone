# Zepto Support Assistant

This module implements an offline Zepto Support Assistant using policy documents, LangGraph, ChromaDB, and FastAPI.

## Features

* Retrieves relevant information from Zepto policy documents.
* Uses ChromaDB for document retrieval.
* Uses LangGraph to manage the question-answering workflow.
* Returns answers along with source documents.
* Provides a confidence score for the response.
* Runs as a FastAPI service.

## Workflow

```text
User Question
      ↓
Policy Document Retrieval
      ↓
ChromaDB
      ↓
LangGraph Workflow
      ↓
Answer Generation
      ↓
Answer + Sources + Confidence
```

## Purpose

The purpose of this module is to demonstrate how an AI-powered support assistant can retrieve relevant information from policy documents and provide grounded answers instead of relying only on general model knowledge.

## Technologies

* Python
* LangGraph
* ChromaDB
* FastAPI
* Policy Documents
* Large Language Model

## Running the Module

The support assistant can be started through the FastAPI application provided in this module.

The API accepts a user support question and returns the generated answer, relevant source documents, and confidence information.
