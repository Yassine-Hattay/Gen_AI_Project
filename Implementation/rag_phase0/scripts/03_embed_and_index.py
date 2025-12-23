import psycopg2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG

CHUNKS_PATH = "data/chunks.pkl"

# Load chunks
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# IDs from step 01
RESOURCE_ID = 1
CONCEPT_ID = 1

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))  # FAISS requires float32

# Save FAISS index
os.makedirs("faiss", exist_ok=True)
faiss.write_index(index, "faiss/index.faiss")

# Insert into PostgreSQL
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

for i, chunk in enumerate(chunks):
    cur.execute("""
        INSERT INTO chunks
        (resource_id, concept_id, chunk_type, difficulty, faiss_index, text)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        RESOURCE_ID,
        CONCEPT_ID,
        chunk["chunk_type"],
        chunk["difficulty"],
        i,
        chunk["text"]  # <-- new
    ))


conn.commit()
conn.close()

print("FAISS + PostgreSQL populated")
