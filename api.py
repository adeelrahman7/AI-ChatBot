from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import predict

app = FastAPI()

class Query(BaseModel):
    sentence: str

@app.get("/")
def root():
    return {"message": "Chatbot API is running."}

@app.post("/chat")
def chat (query: Query):
    response, confidence = predict(query.sentence)
    return {"response": response, "confidence": confidence}

