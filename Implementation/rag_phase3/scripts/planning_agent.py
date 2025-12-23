# scripts/planning_agent.py
# Graph-based learning path

import random

EPSILON = 0.2  # exploration rate

PREREQUISITES = {
    "algebra": [],
    "quadratic_eq": ["algebra"],
    "calculus": ["algebra", "quadratic_eq"],
    "physics": ["calculus"],
    "chemistry": ["algebra"]
}

def generate_plan(concept, depth=2):
    plan = [concept]
    current = concept
    for _ in range(depth - 1):
        prereqs = PREREQUISITES.get(current, [])
        if prereqs:
            current = prereqs[0]
            plan.append(current)
    return plan


def plan_learning_path(profile, state=None):
    knowledge = profile["knowledge"]
    sorted_concepts = [c for c in sorted(knowledge, key=knowledge.get) if c in knowledge]

    # ε-greedy exploration
    if random.random() < EPSILON:
        chosen = random.choice(sorted_concepts)
        mode = "explore"
    else:
        chosen = sorted_concepts[0]
        mode = "exploit"

    # Backtracking if reward is bad
    backtrack = False
    if state and state.get("reward", 0) < -0.05 and state.get("plan_history"):
        # backtrack to PREVIOUS different concept
        for prev in reversed(state["plan_history"][:-1]):
            if prev != sorted_concepts[0]:
                chosen = prev
                break
        else:
            chosen = sorted_concepts[0]  # fallback

        backtrack = True

    print("\n[PLANNING AGENT]")
    print("  Knowledge state:", knowledge)
    print("  Sorted concepts (low → high mastery):", sorted_concepts)
    print("  Mode:", mode)
    print("  Reward received:", state.get("reward") if state else None)

    if backtrack:
        print("  ⚠️ BACKTRACKING TRIGGERED")
        print("  ↩ Returning to previous concept:", chosen)
    else:
        print("  ➡ Selected concept:", chosen)

    print("  Planned horizon:", generate_plan(chosen))

    return {
        "current_concept": chosen,
        "plan": generate_plan(chosen),
        "mode": mode,
        "backtrack": backtrack
    }


def plan_path(profile, state=None):
    return plan_learning_path(profile, state)


def generate_plan(concept, depth=2):
    plan = [concept]
    for _ in range(depth - 1):
        prereqs = PREREQUISITES.get(concept, [])
        if prereqs:
            concept = prereqs[0]
            plan.append(concept)
    return plan
