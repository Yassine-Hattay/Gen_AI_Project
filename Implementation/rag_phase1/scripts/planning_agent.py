def plan_learning_path(profile, concepts):
    weakest = min(profile["knowledge"], key=profile["knowledge"].get)
    return {"next_concept": weakest, "recommendation_type": "example"}
