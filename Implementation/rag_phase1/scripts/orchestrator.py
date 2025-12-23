from profile_agent import profile_learner
from planning_agent import plan_learning_path
from content_generator import generate_content
from xai_agent import justify_recommendation

def orchestrator(quiz_scores, concepts):
    profile = profile_learner(quiz_scores)
    plan = plan_learning_path(profile, concepts)
    content = generate_content(plan, profile)
    explanation = justify_recommendation(plan, profile)
    return content, explanation

if __name__ == "__main__":
    quiz_scores = {"linear_equations": 0.9, "quadratic_equations": 0.3}
    concepts = ["linear_equations", "quadratic_equations"]

    content, explanation = orchestrator(quiz_scores, concepts)

    print("=== Recommended Content ===")
    print(content)
    print("\n=== Explanation ===")
    print(explanation)
