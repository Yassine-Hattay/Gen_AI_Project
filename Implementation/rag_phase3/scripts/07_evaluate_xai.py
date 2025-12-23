# scripts/07_evaluate_xai.py
import psycopg2

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Evaluate recommendation scores
cur.execute("SELECT AVG(final_score) FROM recommendations")
hybrid_avg = cur.fetchone()[0]

cur.execute("SELECT AVG(random()) FROM recommendations")
random_avg = cur.fetchone()[0]

# Evaluate explanation length & richness
cur.execute("SELECT explanation FROM recommendations WHERE explanation IS NOT NULL")
explanations = cur.fetchall()
avg_length = sum(len(exp[0].split()) for exp in explanations) / len(explanations)

print("Hybrid avg score:", hybrid_avg)
print("Random baseline:", random_avg)
print("Avg explanation length (words):", avg_length)
conn.close()
