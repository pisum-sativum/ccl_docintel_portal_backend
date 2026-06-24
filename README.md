# CCL Document Intelligence Portal (Backend)

This repository contains the backend service for the Central Coalfields Limited (CCL) Document Intelligence Portal. It is a Python-based API built with FastAPI that provides document processing, compliance scanning, and natural language retrieval capabilities.

## Architecture

The backend operates as a RESTful API and handles heavy computational tasks, including vector embeddings and AI compliance generation.

- Framework: FastAPI
- Database (Relational): Neon Cloud PostgreSQL (via SQLAlchemy)
- Database (Vector): ChromaDB (Local embedded)
- AI Integration: Google Gemini API (Text and Embeddings)
- Authentication: JWT with Passlib (Bcrypt)

## Features

- Document Ingestion: Parses text from PDF, DOCX, TXT, and image files using PyMuPDF, python-docx, and OCR.
- AI Compliance Engine: Automatically scans ingested documents to detect operational and regulatory compliance risks.
- Semantic RAG Pipeline: Vectorizes document chunks into ChromaDB, enabling high-accuracy semantic search and AI chatbot capabilities.
- Security: Granular Role-Based Access Control (RBAC) ensuring appropriate data partitioning between Public, Internal, and Confidential documents.
- Deduplication: Smart hashing and fuzzy semantic matching algorithms to reject redundant uploads.

## Development Setup

1. Install Python (v3.10 or higher recommended).
2. Clone this repository and navigate to the project root.
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   # source .venv/bin/activate    # On Mac/Linux
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a `.env` file in the root directory and configure your keys:
   ```env
   DATABASE_URL=postgresql://<user>:<password>@<host>/<dbname>?sslmode=require
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
6. Start the FastAPI development server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
7. The API documentation (Swagger UI) will be available at http://localhost:8000/docs.

## Project Structure

- `main.py`: Core FastAPI application and endpoint definitions.
- `database.py`: SQLAlchemy models, schema definitions, and connection handlers.
- `auth.py`: JWT token generation, password hashing, and dependency injection for secured routes.
- `extractor.py`: Utility functions for parsing raw text from various file formats.
- `rag_engine.py`: Logic for embedding generation, ChromaDB vector indexing, and RAG chat interactions.
- `documents/`: Local directory for temporarily staging uploaded files before processing.
