import psycopg2
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

styles = ["visual", "textual", "problem-solving"]

def knowledge_to_level(knowledge_score):
    """Convert numeric knowledge (0-1) to textual level"""
    if knowledge_score < 0.4:
        return "beginner"
    elif knowledge_score < 0.7:
        return "intermediate"
    else:
        return "advanced"

# Connect to DB
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Ensure embedding column exists
cur.execute("""
ALTER TABLE learners
ADD COLUMN IF NOT EXISTS embedding FLOAT8[];
""")
conn.commit()

# Option 1: Insert new learners
for _ in range(15):
    style = random.choice(styles)
    knowledge_score = round(random.uniform(0.2, 0.9), 2)
    level = knowledge_to_level(knowledge_score)

    # Generate embedding
    emb = model.encode([f"{style} learner math level {knowledge_score}"])[0]

    cur.execute("""
        INSERT INTO learners (learning_style, level, embedding)
        VALUES (%s, %s, %s)
    """, (style, level, emb.tolist()))

# Option 2: Update existing learners (if you want to populate embeddings for them)
cur.execute("SELECT id, learning_style, level FROM learners WHERE embedding IS NULL")
for learner_id, style, level in cur.fetchall():
    knowledge_score = {"beginner": 0.3, "intermediate": 0.55, "advanced": 0.8}[level]
    emb = model.encode([f"{style} learner math level {knowledge_score}"])[0]
    cur.execute(
        "UPDATE learners SET embedding = %s WHERE id = %s",
        (emb.tolist(), learner_id)
    )

conn.commit()
conn.close()

print("Simulated learners inserted/updated with embeddings")
