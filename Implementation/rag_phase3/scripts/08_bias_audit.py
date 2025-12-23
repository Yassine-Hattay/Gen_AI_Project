# scripts/08_bias_audit.py
import psycopg2
import os
import sys

# Add project root to path if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG  # your database connection settings

# Connect to the database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Query: average recommendation scores by learning style
cur.execute("""
SELECT l.learning_style, AVG(r.final_score)
FROM recommendations r
JOIN learners l ON r.learner_id = l.id
GROUP BY l.learning_style
""")

bias_stats = cur.fetchall()
print("Avg final scores by learning style:", bias_stats)

# Close connection
cur.close()
conn.close()
