# importing main web framework (fastapi), and libraries for handling file uploads,
# sending proper HTTP error responses, and enabling CORS (allowing comms between frontend and backend)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# handle requests, file extraction, hashing and timestamps
import requests
import os
import PyPDF2
from pptx import Presentation
import docx
from io import BytesIO
import hashlib
from datetime import datetime

# creating the FastAPI app 
app = FastAPI(
    title = "AI Study Helper",
    description = "An AI-powered chatbot to assist with study materials.",
    version = "1.0.0"
)

# configuring CORS - allowing any frontend to call the api without restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# loading the embedder, a place to store the uploaded docs
# and a directory for storing the uploaded files
embedder = None
documents_db = {}
UPLOAD_DIR = "uploads"   
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models for request bodies (Chat requests, flashcards, questions, study materials, document metadata)
# these define the structure of the data we expect in requests and return in responses

# client MUST send a document_id and material_type
class StudyMaterialRequest(BaseModel):
    document_id: str
    material_type: str  # e.g., "pdf", "pptx", "docx"
    topic : Optional[str] = None

class ChatRequest(BaseModel):
    documents_id: str
    question: str

class FlashCard(BaseModel):
    front: str
    back: str
    topic: str

class Question(BaseModel):
    question: str
    type: str  # e.g., "multiple_choice", "short_answer"
    options: Optional[List[str]] = None
    answer: str
    explanation: str

class StudyMaterial(BaseModel):
    document_id: str
    material_type: str
    content: str
    created_at: str

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_date: str
    material_type: str
    page_count: str
    text_length: str
    topics: List[str]

# startup event to load the embedder model
@app.on_event("startup")
async def startup_event():
    global embedder
    print("Loading embedder model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedder model loaded.")

# helper functions - converting binary files into raw text (critical for embedding and indexing)
# Extract text from PDF files
def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Extract text from pptx files
def extract_text_from_pptx(file_bytes: bytes) -> str:
    prs = Presentation(BytesIO(file_bytes))
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

# Extract text from docx files
def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(BytesIO(file_bytes))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to chunk text into smaller pieces
# LLMs and embedding models have token limits, so we need to split large texts

# This function splits text into chunks of a specified size (in characters) 
# prevents overly large embedding inputs
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for the space
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# Function to create a FAISS index from text chunks
# this converts texts to embeddings, to float32, normalizes the vectors, and creates the FAISS index and adds vectors to the index.
# this enables semantic similarity search (CORE RAG ARCHITECTURE)
def create_faiss_index(chunks: List[str]):
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

# Sends prompt to Ollama API and retrieves AI-generated response, handles timeouts and errors
# Allows us to generate summaries, flashcards, questions, and chat responses
def generate_with_ollama(prompt: str, model: str = "llama2") -> str:        
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options':{
                    'temperature':0.7,
                    'num_predict':1000
                }
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "Error: Ollama unavailable."
    except Exception as e:
        return f"Error: {str(e)}"

# Function to extract topics from text using AI
# Analyzes doc, extracts 3-5 main topics, and returns them as a list
# Used for metadata, flashcards, and question generation
def extract_topics(text: str) -> List[str]:
    """Extract main topics from text (temporary - no AI)."""
    # Simple keyword extraction until Ollama is set up
    keywords = ['biology', 'chemistry', 'physics', 'math', 'history', 
                'science', 'anatomy', 'cell', 'molecule', 'equation']
    
    text_lower = text.lower()
    found_topics = []
    
    for keyword in keywords:
        if keyword in text_lower:
            found_topics.append(keyword.capitalize())
    
    # Return first 3 found topics, or generic ones
    if found_topics:
        return found_topics[:3]
    else:
        return ["General Study Notes", "Education", "Learning"]


# Uncomment below to use AI-based topic extraction when Ollama is set up

#     """Extract main topics from text using AI."""
#     prompt = f"""Analyze this text and extract 3-5 main topics/subjects covered.
# Return ONLY a comma-separated list of topics, nothing else.

# Text:
# {text[:2000]}

# Topics:"""

#     response = generate_with_ollama(prompt)
#     topics = [t.strip() for t in response.split(',')]
#     return topics[5]

# Root endpoint - basic info about the API
# confirms the API is running and provides version and feature list
@app.get("/")
def root():
    return {"message": "Welcome to the AI Study Helper API!"
            "version 1.0.0",
            "features": [
                "Upload study materials (PDF, PPTX, DOCX)",
                "Generate summary",
                "Create flashcards",
                "Generate practice questions",
                "Chat with your study materials"
            ]
        }

# Endpoint to upload a document
# Validates file type, extracts text, chunks it, creates FAISS index, extracts topics, and stores metadata
# This is the DOCUMENT UPLOAD AND PROCESSING CORE
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a study material document (PDF, PPTX, DOCX).
    Returns document ID for further processing.
    """
    
    #validating file type
    allowed_extensions = ['.pdf', '.pptx', '.docx']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail= f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )
    
    # try-catch statements to handle file reading and text extraction
    try: 
        file_bytes = await file.read()
        
        if file_ext == '.pdf':
            text = extract_text_from_pdf(file_bytes)
        elif file_ext == '.pptx':
            text = extract_text_from_pptx(file_bytes)
        elif file_ext == '.docx':
            text = extract_text_from_docx(file_bytes)
        
        if not text.strip():
            raise HTTPException(
                status_code=400, detail="No extractable text found in the document."
            )
        
        # creating a doc id, chunking text, creating faiss index, extracting topics
        doc_id = hashlib.md5(file_bytes).hexdigest()
        chunks = chunk_text(text, chunk_size=500)
        index, embeddings = create_faiss_index(chunks)
        topics = extract_topics(text)
        
        # storing document info in the in-memory db
        documents_db[doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "upload_date": datetime.utcnow().isoformat(),
            "text": text,
            "chunks": chunks,
            "index": index,
            "embeddings": embeddings,
            "topics": topics,
            "page_count": len(chunks),
            "text_length": len(text),
        }
        
        return {"document_id": doc_id,
                "filename": file.filename,
                "pages_processed": len(chunks),
                "topics_found": topics,
                "message": "Document uploaded and processed successfully."
                }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )

# Listing uploaded documents endpoint
# returns metadata for all uploaded documents
@app.get("/documents")
def list_documents():
    """
    List all uploaded documents with their metadata.
    """
    docs = []
    for doc_id in documents_db.items():
        docs.append({
            "id": doc_id,
            "filename": doc["filename"],
            "upload_date": doc["upload_date"],
            "page_count": doc["page_count"],
            "topics": doc["topics"],
        })
    return {"documents": docs, "total": len(docs)}

# Endpoint to generate a summary for a document
# takes document id, retrieves text, constructs prompt, calls LLM, and returns summary
@app.post("/generate/summary")
async def generate_summary(request: StudyMaterialRequest):
    """ Generate a summary for the specified document."""
    
    if request.document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    doc = documents_db[request.document_id]
    text = doc["text"][:3000] # limit to first 3000 chars for prompt size
    
    prompt = f"""Summarize the following study material in a concise manner. Include:
1. Main topics covered
2. Key concepts and definitions
3. Important facts, events, or formulas

Format as clear bullet points.
Content:
{text}
    
Summary:"""
    summary = generate_with_ollama(prompt)
    
    return {"document_id": request.document_id,
            "material_type": "summary",
            "content": summary,
            "created_at": datetime.now().isoformat()
            }

# Endpoint to generate flashcards for a document
# prompts LLM to create flashcards, parses response, and returns structured flashcard data, assigning topics automatically
@app.post("/generate/flashcards")
async def generate_flashcards(request: StudyMaterialRequest):
    """ Generate flashcards for the specified document."""
    
    if request.document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    doc = documents_db[request.document_id]
    text = doc["text"][:3000] # limit to first 3000 chars for prompt size
    
    prompt = f"""Create 10 flashcards based on the following study material.
Format EXACTLY as:
FRONT: [question or term]
BACK: [answer or definition]
---
(repeat for each flashcard)

Content:
{text}

Flashcards:"""
    response = generate_with_ollama(prompt)
    
    # parsing the response into flashcard objects
    cards = []
    cards_texts = response.split('---')
    for card_text in cards_texts:
        if "FRONT:" in card_text and "BACK:" in card_text:
            lines = card_text.strip().split('\n')
            front = ""
            back = ""
            for line in lines:
                if line.startswith("FRONT:"):
                    front = line.replace("FRONT:", "").strip()
                elif line.startswith("BACK:"):
                    back = line.replace("BACK:", "").strip()
                    
            if front and back:
                cards.append({
                    "front": front,
                    "back": back,
                    "topic": doc["topics"][0] if doc["topics"] else "General"
                })
                
    return {"document_id": request.document_id,
            "material_type": "flashcards",
            "flashcards": cards,
            "count": len(cards),
            "created_at": datetime.now().isoformat()
            }
    
# Endpoint to generate practice questions for a document
# requests LLM to create questions, parses response, and returns structured question data
@app.post("/generate/questions")
async def generate_questions(request: StudyMaterialRequest):
    """ Generate practice questions for the specified document."""
    
    if request.document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    doc = documents_db[request.document_id]
    text = doc["text"][:3000] # limit to first 3000 chars for prompt size
    
    prompt = f"""Create 5 practice questions based on the following study material.
Format EXACTLY as:
Q: [question]
TYPE: [multiple_choice/short_answer]
OPTIONS: [A) option1, B) option2, C) option3, D) option4] (only for multiple_choice)
ANSWER: [correct answer]
EXPLANATION: [brief explanation of the answer, why it's correct]
---
(repeat for each question)

Content:
{text}

Questions:"""
    response = generate_with_ollama(prompt)
    
    # parsing the response into question objects
    questions = []
    q_texts = response.split('---')
    for q_text in q_texts[:5]:  # limit to first 5 questions
        if "Q:" in q_text and "TYPE:" in q_text and "ANSWER:" in q_text:
            questions.append({
                "question": "Sample question from your document",
                "type": "multiple_choice",
                "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "answer": "A",
                "explanation": "Based on the content provided..."
            })
            
    return {"document_id": request.document_id,
            "material_type": "questions",
            "questions": questions,
            "count": len(questions),
            "created_at": datetime.now().isoformat()
            }

# Chat endpoint to interact with uploaded documents (RAG)
# Embedding-based retrieval of relevant chunks, constructs chat prompt, calls LLM, and returns response
@app.post("/chat")
async def chat_with_document(request: ChatRequest):
    """ Ask questions about the uploaded document."""
    
    if request.documents_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    doc = documents_db[request.documents_id]
    
    # search for relevant chunks using FAISS
    query_vector = embedder.encode([request.question], convert_to_tensor=False)
    query_vector = np.array(query_vector).astype("float32")
    faiss.normalize_L2(query_vector)
    
    scores, indices = doc["index"].search(query_vector, k=3)  # retrieve top 3 relevant chunks
    
    # get relevant text chunks/context
    context = "\n\n".join([doc["chunks"][idx] for idx in indices[0]])
    
    # generate an answer using the context
    prompt = f"""Use the following context from the study material to answer the question.
Context:
{context}

Question: {request.question}
Provide a clear, educational answer. If the context does not contain the answer, say so.

Answer:"""
    answer = generate_with_ollama(prompt)
    
    return {"document_id": request.documents_id,
            "question": request.question,
            "answer": answer,
            "confidence_score": float(scores[0][0]),
            "sources_used": len(indices[0])
            }

# Endpoint to delete an uploaded document
# removes document and its data from in-memory db
@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """ Delete an uploaded document and its data."""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    filename = documents_db[document_id]["filename"]
    del documents_db[document_id]
    
    return {"message": f"Document '{filename}' and its data have been deleted."}

# To run the app, use the command:
# uvicorn api:app --host
# This will start the FastAPI server on the specified host and port.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
