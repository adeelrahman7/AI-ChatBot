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
from torch.nn.functional import cosine_similarity

# ---------------- NLTK ----------------
nltk.download("punkt")

# ---------------- PATHS ----------------
EMBEDDINGS_PATH = "embeddings.pt"
META_PATH = "embeddings_meta.json"

# ---------------- MODEL ----------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        return self.l2(x)

# ---------------- UTIL ----------------
def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# ---------------- LOAD DATA ----------------
with open("intents.json") as f:
    data = json.load(f)

# ---------------- LOAD OR TRAIN ----------------
# Sentence embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# sentences = []
# sentence_tags = []

# for intent in data["intents"]:
#     tag = intent["tag"]
#     for pattern in intent["patterns"]:
#         sentences.append(pattern)
#         sentence_tags.append(tag)

# sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)


if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(META_PATH):
    # checkpoint = torch.load(DATA_PATH)
    # all_words = checkpoint["all_words"]
    # tags = checkpoint["tags"]

    # model = NeuralNet(len(all_words), 8, len(tags))
    # model.load_state_dict(torch.load(MODEL_PATH))
    # model.eval()
    
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
    
    print("💾 Model trained and saved!")

#     all_words = []
#     tags = []
#     xy = []

#     for intent in data["intents"]:
#         tag = intent["tag"]
#         tags.append(tag)

#         for pattern in intent["patterns"]:
#             words = word_tokenize(pattern.lower())
#             all_words.extend(words)
#             xy.append((words, tag))

#     ignore_words = ["?", "!", ".", ","]
#     all_words = sorted(set(w for w in all_words if w not in ignore_words))
#     tags = sorted(set(tags))

#     X_train = []
#     y_train = []

#     for (pattern_words, tag) in xy:
#         bag = bag_of_words(pattern_words, all_words)
#         X_train.append(bag)
#         y_train.append(tags.index(tag))

#     X_train = torch.from_numpy(np.array(X_train))
#     y_train = torch.tensor(y_train)

#     model = NeuralNet(len(all_words), 8, len(tags))

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     for epoch in range(200):
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     torch.save(model.state_dict(), MODEL_PATH)
#     torch.save({"all_words": all_words, "tags": tags}, DATA_PATH)

#print("💾 Model trained and saved!")

# ---------------- FAISS ----------------
embedding_dim = sentence_embeddings.shape[1]

faiss_embeddings = sentence_embeddings.cpu().numpy().astype('float32')
faiss.normalize_L2(faiss_embeddings)

index = faiss.IndexFlatIP(embedding_dim)
index.add(faiss_embeddings)


# ---------------- CHAT LOOP ----------------
print("\n🤖 Chatbot is ready! Type 'bye' to exit.\n")

while True:
    sentence = input("You: ")
    if sentence.lower() == "bye":
        break

    # PREDICTION WITH SENTENCE EMBEDDINGS
    user_embedding = embedder.encode(sentence)
    user_embedding = np.array([user_embedding]).astype('float32')
    faiss.normalize_L2(user_embedding)
    
    scores, indices = index.search(user_embedding, k=1)
    
    confidence = scores[0][0]
    best_idx = indices[0][0]
    predicted_tag = sentence_tags[best_idx]
    
    # scores = cosine_similarity(user_embedding.unsqueeze(0), sentence_embeddings)
    # best_score, best_idx = torch.max(scores, dim=0)
    
    # predicted_tag = sentence_tags[best_idx.item()]
    # confidence = best_score.item()
    
        # PREDICTION WITHOUT SENTENCE EMBEDDINGS
    # words = word_tokenize(sentence.lower())
    # bag = bag_of_words(words, all_words)
    # bag = torch.tensor(bag).unsqueeze(0)  # ADD BATCH DIMENSION

    # with torch.no_grad():
    #     output = model(bag)
    #     probs = torch.softmax(output, dim=1)

    # confidence, predicted = torch.max(probs, dim=1)
    # confidence = confidence.item()
    # predicted_tag = tags[predicted.item()]

    print(f"🔍 Confidence: {confidence:.2f}")

    if confidence < 0.60:
        predicted_tag = "unknown"
        
    # RESPOND
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            print("Bot:", random.choice(intent["responses"]))
            break   
        
