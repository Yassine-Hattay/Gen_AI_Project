# scripts/chunk_assumptions.py
import random

CONCEPTS = ["algebra", "quadratic_eq", "calculus", "physics", "chemistry"]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]
MODALITIES = ["visual", "textual", "problem-solving"]
LENGTHS = ["short", "medium", "long"]

# Global seed for reproducibility
SEED = 42
random.seed(SEED)

def assume_chunk_metadata(chunk_id: int = None):
    """
    Random chunk metadata controlled by global SEED.
    The chunk_id is ignored for randomness; all randomness comes from SEED.
    """
    return {
        "concept": random.choice(CONCEPTS),
        "difficulty": random.choice(DIFFICULTIES),
        "modality": random.choice(MODALITIES),
        "length": random.choice(LENGTHS)
    }
