# scripts/profiling_agent.py
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import DB_CONFIG

model = SentenceTransformer("all-MiniLM-L6-v2")
styles = ["visual", "textual", "problem-solving"]
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

def knowledge_to_level(knowledge_score):
    if knowledge_score < 0.4:
        return "beginner"
    elif knowledge_score < 0.7:
        return "intermediate"
    else:
        return "advanced"


import random

PROFILES = [
    {
        "learning_style": "visual",
        "preferences": {
            "modality": "visual",
            "difficulty": "beginner",
            "verbosity": "short"
        },
        "knowledge": {
            "algebra": 0.2,
            "quadratic_eq": 0.1,
            "calculus": 0.0
        },
        "goal": "intuition",
        "background": "science"
    },
    {
        "learning_style": "textual",
        "preferences": {
            "modality": "textual",
            "difficulty": "intermediate",
            "verbosity": "detailed"
        },
        "knowledge": {
            "algebra": 0.7,
            "quadratic_eq": 0.5,
            "calculus": 0.2
        },
        "goal": "exam-prep",
        "background": "engineering"
    },
    {
        "learning_style": "problem-solving",
        "preferences": {
            "modality": "problem-solving",
            "difficulty": "advanced",
            "verbosity": "short"
        },
        "knowledge": {
            "algebra": 0.9,
            "quadratic_eq": 0.8,
            "calculus": 0.6
        },
        "goal": "fast-review",
        "background": "math"
    }
]


def get_profile(learner_id):
    # Map IDs to specific profiles for testing
    test_profiles = {
        1: PROFILES[0],  # low knowledge / beginner
        2: PROFILES[1],  # medium
        3: PROFILES[2],  # high
        9999: {          # custom profile to force backtracking
            "learning_style": "visual",
            "preferences": {"modality": "problem-solving", "difficulty": "advanced", "verbosity": "short"},
            "knowledge": {"algebra": 0.0, "quadratic_eq": 0.0, "calculus": 0.0},
            "goal": "intuition",
            "background": "science"
        }
    }

    profile = test_profiles.get(learner_id, test_profiles[9999])

    profile = { "id": learner_id, **profile }

    print("\n[PROFILING AGENT]")
    print("  Simulated profile:", profile)

    return profile
