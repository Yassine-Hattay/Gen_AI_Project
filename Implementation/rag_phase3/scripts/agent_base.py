# scripts/agent_base.py
from abc import ABC, abstractmethod

class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Takes the global shared state and returns an updated state.
        """
        pass
