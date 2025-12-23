# scripts/agents.py
from agent_base import Agent
from profiling_agent import get_profile
from planning_agent import plan_path
from content_generator import generate_content
from recommendation_agent import recommend
from xai_agent import justify
import numpy as np

shap_values = np.load("shap_values.npy", allow_pickle=True)


class ProfilingAgent(Agent):
    name = "profiling"

    def run(self, state):
        state["profile"] = get_profile(state["learner_id"])
        return state


class PlanningAgent(Agent):
    name = "planning"

    def run(self, state):
        state["plan"] = plan_path(state["profile"], state)
        return state


class ContentAgent(Agent):
    name = "content"

    def run(self, state):
        state["content"] = generate_content(
            state["plan"]["current_concept"],
            state["profile"]
        )
        return state


class RecommendationAgent(Agent):
    name = "recommendation"

    def run(self, state):
        state["recommendations"] = recommend(
            state["plan"]["current_concept"],
            state["profile"],
            state["learner_id"]
        )
        
        prev = state.get("prev_score", 0.0)
        current = np.mean([r["final_score"] for r in state["recommendations"]])

        state["reward"] = current - prev
        state["prev_score"] = current
        
        print("\n[RECOMMENDATION AGENT]")
        print("  Previous avg score:", prev)
        print("  Current avg score:", current)
        print("  Reward signal:", state["reward"])
    
        return state


class XAIExplainAgent(Agent):
    name = "xai"

    def run(self, state):
        state["explanation"] = justify(
            state["plan"]["current_concept"],
            state["recommendations"],
            state["profile"],
            shap_values
        )
        return state
