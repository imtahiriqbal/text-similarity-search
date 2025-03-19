# pip install openai faiss-cpu numpy

import openai
import numpy as np
import os
from dotenv import load_dotenv
# Importing faiss library (Facebook AI Similarity Search)
import faiss

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.environ["API_KEY"]

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return np.array(response["data"][0]["embedding"])

# Example sentences
sentences = [
    "I love programming.",
    "Machine learning is fascinating.",
    "Deep learning is a subset of machine learning.",
    "I enjoy writing code in Python."
]

# Generate embeddings for each sentence
embeddings = [get_embedding(sentence) for sentence in sentences]

# Convert embeddings to NumPy array for efficient processing
embeddings = np.array(embeddings)

# Create FAISS index for similarity search (L2 or Cosine)
d = embeddings.shape[1]  # Vector dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings)  # Store embeddings in FAISS

# Search for the closest sentence to a query
# query = "I like coding in Python."
query = "I'm interested in neural networks."
query_embedding = get_embedding(query).reshape(1, -1)  # Get embedding

# Perform search
k = 2  # Get top 2 closest matches
distances, indices = index.search(query_embedding, k) # L2 distance (squared Euclidean distance) and index of 'k' most closest k-nearest sentence

def cosine_similarity(sentence, query):
    # Compute similarity (Cosine Similarity)
    similarity = np.dot(sentence, query) / (np.linalg.norm(sentence) * np.linalg.norm(query))
    print(f"Similarity Score: {similarity:.4f}")

# Print most similar sentences with distance
print("\n🔍 Most Similar Sentences with Distance:")
for i, j in zip(distances[0], indices[0]):
    print(f"- {sentences[j]} (L2 distance: {i:.4f})")
    cosine_similarity(get_embedding(sentences[j]), get_embedding(query))
    
# Less L2 distance and high Similarity score indicates more similarity
