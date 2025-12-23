import numpy as np

def profile_learner(quiz_scores):
    profile = {
        "knowledge": quiz_scores,
        "learning_style": "visual",
        "preferences": ["examples", "exercises"]
    }
    return profile
    