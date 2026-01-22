# MCAT Study Assistant - AI Chatbot

An intelligent MCAT study assistant using semantic search and RAG (Retrieval-Augmented Generation).

## Features
- Semantic search across 40+ physics concepts
- FAISS vector database for fast similarity matching
- Learns from user feedback
- MCAT-level physics, chemistry, and biology support

## Tech Stack
- Python 3.14
- FAISS (vector database)
- SentenceTransformers (embeddings)
- Ollama/Claude (LLM generation)
- FastAPI (backend)

## Evolution
**v1.0** - Basic intent-based chatbot (greetings/farewells)
**v2.0** - Semantic search with MCAT physics knowledge base ← Current
**v3.0** - Full RAG with LLM integration ← In Progress

## Installation
```bash
pip install -r requirements.txt
python chatbot.py
```

## Usage
```
You: What is Newton's second law?
Bot: Newton's Second Law: F = ma, where F is the net force...
```
