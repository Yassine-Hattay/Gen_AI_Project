import psycopg2
import random

import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("SELECT id FROM learners")
learners = [r[0] for r in cur.fetchall()]

cur.execute("SELECT id FROM chunks")
chunks = [r[0] for r in cur.fetchall()]

for learner in learners:
    for _ in range(random.randint(5, 15)):
        cur.execute("""
            INSERT INTO learner_interactions
            (learner_id, chunk_id, interaction_type, rating)
            VALUES (%s, %s, %s, %s)
        """, (
            learner,
            random.choice(chunks),
            random.choice(["view", "complete"]),
            random.randint(1, 5)
        ))

conn.commit()
conn.close()

print("Interactions simulated")
