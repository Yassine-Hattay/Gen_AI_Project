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
SELECT r.id, l.learning_style, r.content_score, r.collaborative_score
FROM recommendations r
JOIN learners l ON r.learner_id = l.id
WHERE r.explanation IS NULL
LIMIT 5
""")

rows = cur.fetchall()

for rid, style, cs, coll in rows:
    explanation = (
        f"This resource was recommended because it matches your {style} learning style "
        f"(content similarity = {cs:.2f}) and was frequently useful to similar learners "
        f"(collaborative score = {coll:.2f})."
    )

    cur.execute("""
        UPDATE recommendations
        SET explanation = %s
        WHERE id = %s
    """, (explanation, rid))

conn.commit()
conn.close()

print("Explanations generated")
