import faiss
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss/index.faiss")

query = "Give an example of quadratic equations"
q_emb = model.encode([query])

D, I = index.search(np.array(q_emb), k=5)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

for idx in I[0]:
    cur.execute("""
        SELECT chunk_type, difficulty
        FROM chunks
        WHERE faiss_index = %s
    """, (int(idx),))
    print(cur.fetchone())

conn.close()
