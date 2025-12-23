# scripts/state_trace.py
def log_step(state, agent_name):
    if "trace" not in state:
        state["trace"] = []

    if agent_name == "planning" and state.get("plan", {}).get("backtrack"):
        state["trace"].append("planning(backtrack)")
    else:
        state["trace"].append(agent_name)

