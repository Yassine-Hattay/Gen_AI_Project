# scripts/orchestrator.py
from state_trace import log_step

from agents import (
    ProfilingAgent,
    PlanningAgent,
    ContentAgent,
    RecommendationAgent,
    XAIExplainAgent
)
from task_graph import TASK_GRAPH


class Orchestrator:
    def __init__(self):
        self.agents = {
            "profiling": ProfilingAgent(),
            "planning": PlanningAgent(),
            "content": ContentAgent(),
            "recommendation": RecommendationAgent(),
            "xai": XAIExplainAgent()
        }

    def run(self, learner_id):
        # Global shared state
        state = {
            "learner_id": learner_id,
            "plan_history": [],
            "reward": -0.95,          
            "step": 0,
            "backtrack_count": 0
        }

        executed = set()
        queue = ["profiling"]
        MAX_STEPS = 3

        while queue:
            agent_name = queue.pop(0)

            # Prevent re-executing agents except planning during backtracking
            if agent_name in executed and not (
                agent_name == "planning" and state.get("plan", {}).get("backtrack")
            ):
                continue

            agent = self.agents[agent_name]
            state = agent.run(state)
            log_step(state, agent_name)
            executed.add(agent_name)

            # ============================
            # 📌 PLANNING STEP
            # ============================
            if agent_name == "planning":
                state["plan_history"].append(state["plan"]["current_concept"])
                state["step"] += 1

                # --- TRACE BACKTRACKING ---
                if state["plan"]["backtrack"]:
                    state["backtrack_count"] += 1
                    print("\n[TRACE]")
                    print("  ↩ BACKTRACKING EVENT")
                    print("  Returned to concept:", state["plan"]["current_concept"])
                    print("  Trace so far:", " -> ".join(state.get("trace", [])))
                else:
                    state["backtrack_count"] = 0

                # 🛑 BACKTRACK SAFETY VALVE
                if state["backtrack_count"] >= 2:
                    print("  🛑 Too many backtracks, forcing forward progress")
                    state["reward"] = 0.0
                    state["plan"]["backtrack"] = False

                print("\n[ORCHESTRATOR]")
                print(f"  Step {state['step']} / {MAX_STEPS}")
                print("  Current concept:", state["plan"]["current_concept"])
                print("  Plan history:", state["plan_history"])
                print("  Mode:", state["plan"]["mode"])
                print("  Backtrack flag:", state["plan"]["backtrack"])

                knowledge = state["profile"]["knowledge"]

                if max(knowledge.values()) > 0.85 or state["step"] >= MAX_STEPS:
                    print("  ✅ Mastery threshold reached or max steps exceeded.")

            # ============================
            # 📌 REWARD FEEDBACK → REPLAN
            # ============================
            if agent_name == "recommendation":
                if state["reward"] < -0.05:
                    print("\n[ORCHESTRATOR]")
                    print("  ⚠️ Negative reward AFTER feedback, scheduling replanning")

                    executed.discard("planning")

                    if not queue or queue[0] != "planning":
                        queue.insert(0, "planning")

            # ============================
            # 📌 TASK GRAPH CONTINUATION
            # ============================
            for next_agent in TASK_GRAPH[agent_name]:
                if next_agent not in executed:
                    queue.append(next_agent)

        print("\n=== STRATEGIC SUMMARY ===")
        print("Plan history:", state.get("plan_history"))
        print("Final reward:", state.get("reward"))
        print("Total planning steps:", state.get("step"))

        return state


if __name__ == "__main__":
    orch = Orchestrator()
    result = orch.run(learner_id=9999)

    print("\n=== AGENT EXECUTION TRACE ===")
    print(" -> ".join(result.get("trace", [])))

    print("\n=== CONTENT ===\n", result.get("content"))
    print("\n=== RECOMMENDATIONS ===\n", result.get("recommendations"))
    print("\n=== EXPLANATION ===\n", result.get("explanation"))
