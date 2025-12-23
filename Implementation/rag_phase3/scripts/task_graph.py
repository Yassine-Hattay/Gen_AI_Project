# scripts/task_graph.py

TASK_GRAPH = {
    "profiling": ["planning"],
    "planning": ["content", "recommendation"],
    "content": ["xai"],
    "recommendation": ["xai"],
    "xai": []
}
