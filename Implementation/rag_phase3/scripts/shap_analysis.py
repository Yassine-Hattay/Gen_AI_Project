# scripts/06_shap_analysis.py
import psycopg2
import numpy as np
import shap
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
sys.path.insert(0, PROJECT_ROOT)
from config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Sample: fetch recommendations and learner embeddings
cur.execute("SELECT r.final_score, l.embedding FROM recommendations r JOIN learners l ON r.learner_id=l.id")
data = cur.fetchall()

X = np.array([row[1] for row in data])  # learner embeddings
y = np.array([row[0] for row in data])  # final_score

# Use a simple linear model as surrogate for SHAP
import xgboost as xgb
model = xgb.XGBRegressor()
model.fit(X, y)

explainer = shap.Explainer(model)
shap_values = explainer(X)

# Save top features for audit
np.save("shap_values.npy", shap_values.values)
print("SHAP analysis complete")
conn.close()
