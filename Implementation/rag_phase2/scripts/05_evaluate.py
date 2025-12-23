import psycopg2

import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("""
SELECT AVG(final_score)
FROM recommendations
""")
hybrid_avg = cur.fetchone()[0]

cur.execute("""
SELECT AVG(random())
FROM recommendations
""")
random_avg = cur.fetchone()[0]

print("Hybrid avg score:", hybrid_avg)
print("Random baseline:", random_avg)
