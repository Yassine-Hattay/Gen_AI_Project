import faiss
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)


from config import DB_CONFIG

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss/index.faiss")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

def generate_content(plan, profile):
    query = f"Give a {profile['preferences'][0]} for {plan['next_concept']}"
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=3)
    
    retrieved_texts = []
    for idx in I[0]:
        cur.execute("SELECT text FROM chunks WHERE faiss_index=%s", (int(idx),))
        row = cur.fetchone()
        if row:
            retrieved_texts.append(row[0])
    
    return "\n\n".join(retrieved_texts)
