from content_generator import generate_content, model, index
import numpy as np

# scripts/xai_agent.py
def justify(next_concept, recommendations, profile, shap_values=None):
    """
    Lightweight, faithful explanation of recommendations.
    Does NOT misuse SHAP.
    """
    top_chunks = sorted(
        recommendations,
        key=lambda x: x["final_score"],
        reverse=True
    )[:3]
    
    prefs = profile["preferences"]

    explanation = (
        f"The system recommended these resources for the concept '{next_concept}' "
        f"because they align with the learner's preferences "
        f"(modality: {prefs['modality']}, difficulty: {prefs['difficulty']}), "
        f"and match assumed pedagogical properties of the content, "
        f"in addition to collaborative popularity signals.\n\n"
    )

    for i, rec in enumerate(top_chunks, 1):
        explanation += (
            f"{i}. Resource chunk {rec['chunk_id']} "
            f"(combined relevance score = {rec['final_score']:.2f})\n"
        )

    explanation += (
        "\nThis explanation is based on a hybrid recommendation strategy "
        "combining content similarity and collaborative popularity."
    )

    return explanation
