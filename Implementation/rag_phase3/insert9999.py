import psycopg2
import numpy as np
from config import DB_CONFIG

# Connect to your database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Create a dummy embedding of size 384
embedding = np.zeros(384, dtype=np.float32).tolist()  # or random values if you want

# Insert learner 9999
cur.execute("""
    INSERT INTO learners (id, embedding, learning_style)
    VALUES (%s, %s, %s)
    ON CONFLICT (id) DO NOTHING
""", (9999, embedding, 'visual'))  # adjust learning_style if needed

conn.commit()
cur.close()
conn.close()

print("Learner 9999 inserted successfully!")
