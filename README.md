# 📚 AI Study Helper

> ⚠️ **Work in Progress** - This project is under active development and may contain bugs.

An intelligent document-based study assistant that transforms your notes, slides, and textbooks into interactive study materials using AI.

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What It Does

Upload any study document (PDF, PowerPoint, Word) and the AI automatically generates:
- ✅ **Summaries** - Concise overviews of key concepts
- ✅ **Flashcards** - Question/answer pairs for memorization
- ✅ **Practice Questions** - Multiple choice and short answer questions
- ✅ **Interactive Chat** - Ask questions about your uploaded materials

---

## 🚀 Features

### Core Functionality
- **📄 Multi-Format Support** - PDF, PPTX, DOCX files
- **🤖 AI-Powered Generation** - Uses Ollama (Llama 3.2) for intelligent content creation
- **💬 RAG Chat System** - Ask questions and get answers from your documents
- **🔍 Semantic Search** - FAISS vector database for fast, accurate retrieval
- **📊 Topic Extraction** - Automatically identifies main subjects in your materials

### Security & Performance
- **🔒 Rate Limiting** - Protection against abuse (5 uploads/min, 10 generations/min, 20 chats/min)
- **✅ Input Validation** - Strict Pydantic models with sanitization
- **🛡️ Prompt Injection Protection** - Sanitizes all LLM inputs
- **📝 Comprehensive Logging** - Track all API activity
- **🔑 API Key Ready** - Optional authentication infrastructure
- **⚡ Optimized Performance** - Efficient chunking and embedding strategies
- 
---

## 📦 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI | REST API server |
| **LLM** | Ollama (Llama 3.2) | AI text generation |
| **Embeddings** | SentenceTransformers | Text vectorization |
| **Vector DB** | FAISS | Semantic similarity search |
| **Document Parsing** | PyPDF2, python-pptx, python-docx | Extract text from files |
| **Security** | slowapi, Pydantic | Rate limiting & validation |
| **Logging** | Python logging | Activity tracking |

---

## 🛠️ Installation

### Prerequisites
- Python 3.14+
- Ollama (for AI generation)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/ai-study-helper.git
cd ai-study-helper
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.26.2
requests==2.31.0
PyPDF2==3.0.1
python-pptx==0.6.23
python-docx==1.1.0
python-multipart==0.0.6
slowapi==0.1.9
python-dotenv==1.0.0
```

### Step 3: Install Ollama
1. Download from [https://ollama.ai](https://ollama.ai)
2. Install the application
3. Download the AI model:
```bash
ollama pull llama3.2
```

### Step 4: Setup Environment (Optional)
Create a `.env` file for API key protection:
```bash
API_KEY_ENABLED=false
API_KEY=your-secret-key-here
```

---

## 🚀 Quick Start

### Start the Backend Server
```bash
uvicorn main:app --reload
```

Server runs at: **http://localhost:8000**

### Access the API Documentation
Open your browser: **http://localhost:8000/docs**

You'll see interactive Swagger UI with all endpoints!

---

## 📖 API Usage Examples

### 1. Upload a Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@lecture_notes.pdf"
```

**Response:**
```json
{
  "document_id": "abc123def456...",
  "filename": "lecture_notes.pdf",
  "pages_processed": 15,
  "topics_found": ["Cell Biology", "Mitosis", "DNA Replication"],
  "message": "Document uploaded and processed successfully."
}
```

### 2. Generate Summary
```bash
curl -X POST "http://localhost:8000/generate/summary" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "abc123def456...",
    "material_type": "summary"
  }'
```

**Response:**
```json
{
  "document_id": "abc123def456...",
  "material_type": "summary",
  "content": "Main topics:\n• Cell division processes\n• DNA replication mechanisms\n...",
  "created_at": "2026-01-24T10:30:00"
}
```

### 3. Generate Flashcards
```bash
curl -X POST "http://localhost:8000/generate/flashcards" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "abc123def456...",
    "material_type": "flashcards"
  }'
```

**Response:**
```json
{
  "document_id": "abc123def456...",
  "material_type": "flashcards",
  "flashcards": [
    {
      "front": "What is mitosis?",
      "back": "Cell division producing two identical daughter cells",
      "topic": "Cell Biology"
    }
  ],
  "count": 10,
  "created_at": "2026-01-24T10:31:00"
}
```

### 4. Chat with Your Document
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "abc123def456...",
    "question": "What happens during prophase?"
  }'
```

**Response:**
```json
{
  "document_id": "abc123def456...",
  "question": "What happens during prophase?",
  "answer": "During prophase, chromatin condenses into chromosomes...",
  "confidence_score": 0.87,
  "sources_used": 3
}
```

### 5. List All Documents
```bash
curl -X GET "http://localhost:8000/documents"
```

### 6. Delete a Document
```bash
curl -X DELETE "http://localhost:8000/documents/abc123def456..."
```

---

## 🔐 Security Features

### Rate Limiting
Prevents API abuse with tiered limits:
- **Uploads:** 5 per minute
- **Generations:** 10 per minute (summaries, flashcards, questions)
- **Chat:** 20 per minute
- **Document List:** 30 per minute

### Input Validation
- ✅ File size limit: 10MB
- ✅ Allowed formats: PDF, PPTX, DOCX only
- ✅ MIME type validation
- ✅ Question length: 3-500 characters
- ✅ Document ID format: exactly 32 characters (MD5 hash)

### Prompt Injection Protection
All text sent to LLM is sanitized:
- Removes code blocks (```)
- Strips HTML/XML tags
- Blocks injection attempts
- Limits context length

---

## 📊 API Endpoints Reference

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `GET` | `/` | Frontend interface | - |
| `GET` | `/health` | API health check | - |
| `POST` | `/upload` | Upload document | 5/min |
| `GET` | `/documents` | List all documents | 30/min |
| `POST` | `/generate/summary` | Generate summary | 10/min |
| `POST` | `/generate/flashcards` | Generate flashcards | 10/min |
| `POST` | `/generate/questions` | Generate questions | 10/min |
| `POST` | `/chat` | Chat with document | 20/min |
| `DELETE` | `/documents/{id}` | Delete document | 10/min |


---

## 🔄 Evolution History

### Version 1.0 - Intent-Based Chatbot
- Basic pattern matching for greetings/farewells
- ~20 hardcoded intents
- Simple response selection

### Version 2.0 - MCAT Semantic Search
- FAISS vector database
- 40+ MCAT physics concepts
- Semantic similarity matching
- User feedback learning system

### Version 3.0 - AI Study Helper (Current)
- **Complete pivot** from Q&A to document processing
- Multi-format document upload (PDF/PPTX/DOCX)
- RAG-powered generation system
- AI-generated study materials
- Production-ready security
- Web interface

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Adeel**
- GitHub: [@adeelrahman7](https://github.com/adeelrahman7)
- Project Link: [https://github.com/adeelrahman7/ai-study-helper](https://github.com/adeelrahman7/AI-Study-Helper.git)

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local LLM infrastructure
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing framework
- [Sentence Transformers](https://www.sbert.net/) for embeddings


**⭐ If this helped you study better, give it a star!**
