import json
import torch
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer


# paths to data files
EMBEDDINGS_PATH = "embeddings.npy"
QUESTIONS_PATH = "questions.json"
FAISS_INDEX_PATH = "faiss.index"

# loading model 
print ("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# loading data 
def load_questions():
    try: 
        with open(QUESTIONS_PATH, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return data["documents"]
    except FileNotFoundError:
        print (f"Error: {QUESTIONS_PATH} not found.")
        return []
    except json.JSONDecodeError:
        print (f"Error: Failed to decode JSON from {QUESTIONS_PATH}.")
        return []


# documents = load_questions()
# texts = [doc["text"] for doc in documents]

# building embeddings
def build_or_load_embeddings(texts):
    if os.path.exists(EMBEDDINGS_PATH):
        print ("Loading embeddings...")
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        print ("Building embeddings...")
        embeddings = embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        np.save(EMBEDDINGS_PATH, embeddings)
        print ("Embeddings built and saved.")
    return embeddings

# def build_embeddings(texts):
#     print ("Building embeddings...")
#     embeddings = embedder.encode(texts, convert_to_tensor=True)
#     torch.save(embeddings, EMBEDDINGS_PATH)
#     return embeddings

# def loading_embeddings():
#     print ("Loading embeddings...")
#     return torch.load(EMBEDDINGS_PATH)
    
# if os.path.exists(EMBEDDINGS_PATH) :
#         embeddings = loading_embeddings()
# else:
#     embeddings = build_embeddings()
        
# building faiss index
def building_faiss_index(embeddings):
    print ("Building FAISS index...")
    embeddings_np = embeddings.copy()
    faiss.normalize_L2(embeddings_np)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print ("FAISS index built and saved.")
    return index

def loading_faiss_index():
    print ("Loading FAISS index...")
    return faiss.read_index(FAISS_INDEX_PATH)

# chat function 
def predict(user_input, index, documents, k=3, confidence_threshold=0.45):
    
    """
    Find the best matching response for user input.
    
    Args:
        user_input: User's question/query
        index: FAISS index
        documents: List of document dictionaries
        k: Number of nearest neighbors to retrieve
        confidence_threshold: Minimum similarity score (0-1)
    
    Returns:
        tuple: (response_text, confidence_score)
    """
    
    query_vector = embedder.encode([user_input], convert_to_tensor=False)
    query_vector = np.array(query_vector).astype('float32')
    
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector,k)

    best_score = float(scores[0][0])
    best_index = int(indices[0][0])
    
    # confidence = float(1/(1 + best_distance))
    # confidence = float(max(0.0, 1 - (1 - best_sim)/2))

    if best_score < confidence_threshold:
        return ("I'm not sure I understand."
                "Try asking another question.", best_score)
        
    best_doc = documents[best_index]
    return best_doc["text"], best_score

# testing section 
def main():
    documents = load_questions()
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    texts = [doc["text"] for doc in documents]
    
    embeddings = build_or_load_embeddings(texts)
    
    if os.path.exists(FAISS_INDEX_PATH):
        index = loading_faiss_index()
    else:
        index = building_faiss_index(embeddings)
        
    print("\n" + "="*50)
    print ( "\n Chatbot is ready! Type 'quit' to exit. \n" )
    print("="*50 + "\n")
    
    while True:
        try: 
            user_input = input("You: ").strip()
            
            if not user_input:
                continue 
            
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting chat. Goodbye!")
                break
            
            response, confidence = predict(user_input, index, documents)
            print(f"Bot: {response} (Confidence: {confidence:.2f})\n")
            
        except KeyboardInterrupt:
            print("\nExiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            
if __name__ == "__main__":
    main()

