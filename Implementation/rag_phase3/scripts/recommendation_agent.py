# scripts/recommendation_agent.py
from chunk_assumptions import assume_chunk_metadata

import psycopg2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import DB_CONFIG

ALPHA = -1
BETA = 0.3
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss/index.faiss")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

def recommend(next_concept, profile, learner_id=None, top_k=5):
    """Hybrid recommendation combining content-based & collaborative filtering"""
    if learner_id is None:
        cur.execute("SELECT id, embedding FROM learners LIMIT 1")
        learner_id, learner_emb = cur.fetchone()
    else:
        # Fetch the embedding for the given learner
        cur.execute("SELECT embedding FROM learners WHERE id=%s", (learner_id,))
        row = cur.fetchone()
        if row is None or row[0] is None:
            raise ValueError(f"No embedding found for learner {learner_id}")
        learner_emb = row[0]

    learner_emb = np.array(learner_emb, dtype=np.float32).reshape(1, -1)

    # content-based
    D, I = index.search(learner_emb.astype("float32"), k=top_k)
    recommendations = []

    for rank, idx in enumerate(I[0]):
        chunk_meta = assume_chunk_metadata(int(idx))

        # collaborative score
        cur.execute("SELECT COUNT(*) FROM learner_interactions WHERE chunk_id=%s", (int(idx),))
        collab_score = float(cur.fetchone()[0])

        final_score = ALPHA + BETA * collab_score

        # learner preferences
        prefs = profile["preferences"]

        # --- ASSUMED CONTENT ALIGNMENT ---
        concept_bonus = 0.0
        if chunk_meta["concept"] == next_concept:
            concept_bonus = 0.25

        difficulty_bonus = 0.0
        if chunk_meta["difficulty"] == prefs["difficulty"]:
            difficulty_bonus = 0.1

        modality_bonus = 0.0
        if chunk_meta["modality"] == prefs["modality"]:
            modality_bonus = 0.1

        # learning-style global boost (unchanged)
        style_bonus = 0.0
        if profile["learning_style"] == "visual":
            style_bonus = 0.1
        elif profile["learning_style"] == "problem-solving":
            style_bonus = 0.15

        final_score = (
            final_score
            + concept_bonus
            + difficulty_bonus
            + modality_bonus
            + style_bonus
        )

        cur.execute("""
                INSERT INTO recommendations
                (learner_id, chunk_id, content_score, collaborative_score, final_score)
                VALUES (%s, %s, %s, %s, %s)
            """, (learner_id, int(idx), 0, collab_score, final_score))


        recommendations.append({"chunk_id": int(idx), "final_score": final_score})

    conn.commit()
    return recommendations
