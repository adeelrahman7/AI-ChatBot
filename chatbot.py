import json
import random
import nltk
import torch
import torch.nn as nn
import numpy as np
import os
import faiss
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer


# ---------------- PATHS ----------------
EMBEDDINGS_PATH = "embeddings.pt"
META_PATH = "embeddings_meta.json"

with open ("intents.json") as f:
    data = json.load(f)
    
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def loading_embeddings():
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(META_PATH):
        sentence_embeddings = torch.load(EMBEDDINGS_PATH)
        
        with open(META_PATH) as f:
            meta = json.load(f)
            sentences = meta["sentences"]
            sentence_tags = meta["sentence_tags"]

        print("✅ Loaded pre-trained model!")

    else:
        print("⚠️ Training new model...")
        
        sentences = []
        sentence_tags = []

        for intent in data["intents"]:
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                sentences.append(pattern)
                sentence_tags.append(tag)

        sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)   
        torch.save(sentence_embeddings, EMBEDDINGS_PATH)
        
        with open(META_PATH, 'w') as f:
            json.dump({
                "sentences": sentences,
                "sentence_tags": sentence_tags
            }, f)   
        
        print("💾 Embeddings and saved!")
    return sentence_embeddings, sentences, sentence_tags

sentence_embeddings, sentences, sentence_tags = loading_embeddings()


# ---------------- FAISS Index Builder ----------------
def loading_faiss_index(sentence_embeddings):
    embeddings = sentence_embeddings.cpu().numpy().astype('float32')
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

index = loading_faiss_index(sentence_embeddings)


def predict(sentence, threshold=0.60):
    # PREDICTION WITH SENTENCE EMBEDDINGS
    user_embedding = embedder.encode(sentence)
    user_embedding = np.array([user_embedding]).astype('float32')
    faiss.normalize_L2(user_embedding)
    
    scores, indices = index.search(user_embedding, k=1)
    confidence = float(scores[0][0])
    best_idx = int(indices[0][0])
    predicted_tag = sentence_tags[best_idx]
    
    if confidence < threshold:
        predicted_tag = "unknown"
        
    # RESPOND
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"]), confidence
            
    return "I'm not sure I understand.", confidence

