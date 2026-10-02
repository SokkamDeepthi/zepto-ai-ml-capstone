# Capstone Project — Certificate Program in Artificial Intelligence and Machine Learning

## Project Overview

This capstone project combines data engineering, data analysis, machine learning, and an AI-powered support assistant into a single end-to-end project.

The project contains three major modules:

1. Data Pipeline and SQL Analytics
2. Titanic Survival Prediction using Machine Learning
3. Zepto Support Assistant using Retrieval-Augmented Policy Search

---

# Module 1 — Data Pipeline and SQL Analytics

## Objective

Build an end-to-end data pipeline that collects book data, cleans the dataset, stores it in a SQLite database, and performs SQL-based analysis.

## Workflow

```text
Web Scraping
     ↓
Data Cleaning
     ↓
CSV Dataset
     ↓
SQLite Database
     ↓
SQL Queries
     ↓
Analysis Output


## 🚀 Live Demo

The Zepto Support Assistant is deployed on Render and available online.

**Live API:**
https://zepto-ai-ml-capstone-1.onrender.com

**Swagger API Documentation:**
https://zepto-ai-ml-capstone-1.onrender.com/docs

### Example API Query

**Question:**
`What is Zepto's refund policy?`

The API returns:

* AI-generated policy answer
* Retrieved source documents
* Confidence score

The application uses FastAPI, ChromaDB, LangGraph, and policy documents to provide a retrieval-based customer support assistant.
