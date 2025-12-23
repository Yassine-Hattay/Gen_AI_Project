import psycopg2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG

ALPHA = 0.7
BETA = 0.3

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("../rag_phase0/faiss/index.faiss")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# pick a learner
cur.execute("SELECT id, embedding FROM learners LIMIT 1")
learner_id, learner_emb = cur.fetchone()
learner_emb = np.array(learner_emb).reshape(1, -1)

# content-based
D, I = index.search(learner_emb.astype("float32"), k=5)

for rank, idx in enumerate(I[0]):
    content_score = float(1 / (1 + D[0][rank]))  # cast to Python float

    # collaborative score
    cur.execute("""
        SELECT COUNT(*)
        FROM learner_interactions
        WHERE chunk_id = %s
    """, (int(idx),))
    collaborative_score = float(cur.fetchone()[0])  # cast to Python float

    final_score = float(ALPHA * content_score + BETA * collaborative_score)  # cast

    cur.execute("""
        INSERT INTO recommendations
        (learner_id, chunk_id, content_score, collaborative_score, final_score)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        learner_id,
        int(idx),
        content_score,
        collaborative_score,
        final_score
    ))

conn.commit()
conn.close()

print("Hybrid recommendations stored")
