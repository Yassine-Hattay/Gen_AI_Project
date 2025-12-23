import os
import sys

# ✅ Add project root (rag_phase0) to Python path FIRST
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import psycopg2
from config import DB_CONFIG


conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("""
INSERT INTO courses (name, domain)
VALUES ('Linear & Quadratic Equations', 'Mathematics')
RETURNING id;
""")
course_id = cur.fetchone()[0]

cur.execute("""
INSERT INTO concepts (name, course_id, difficulty)
VALUES ('quadratic_equations', %s, 'beginner')
RETURNING id;
""", (course_id,))
concept_id = cur.fetchone()[0]

cur.execute("""
INSERT INTO resources (type, source, course_id)
VALUES ('pdf', 'quadratic_eq.pdf', %s)
RETURNING id;
""", (course_id,))
resource_id = cur.fetchone()[0]

conn.commit()
conn.close()

print(course_id, concept_id, resource_id)
