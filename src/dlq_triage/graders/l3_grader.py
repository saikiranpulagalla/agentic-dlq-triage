"""L3 Grader: Cascading failure root cause diagnosis scoring."""

from typing import Dict, Any, List
from dlq_triage.models import Action


class L3Grader:
    """Grades agent actions for L3 cascading failure scenarios."""

    @staticmethod
    def grade(action: Action, scenario: Dict[str, Any]) -> tuple[float, float]:
        """Grade the agent's action for a cascading failure.
        
        Returns (root_cause_score, idempotency_score) separately.
        
        Scoring logic:
        - root_cause_score: +0.99 if root cause correctly identified, 0.01 otherwise
        - idempotency_score: +0.99 if agent correctly chose RETRY knowing tool is non-idempotent, 0.01 otherwise
        
        Args:
            action: The agent's action
            scenario: The failure scenario
            
        Returns:
            Tuple of (root_cause_score, idempotency_score) strictly between 0.0 and 1.0
        """
        try:
            root_cause_score = 0.01  # Minimum valid score
            idempotency_score = 0.01  # Minimum valid score

            # root_cause_score (+0.98 if root cause correctly identified)
            # maps to whether action.root_cause_tool matches scenario root
            if action.root_cause_tool == scenario.get("root_cause_tool"):
                root_cause_score = 0.99  # Maximum valid score (not 1.0)

            # idempotency_score (+0.98 if agent correctly chose RETRY
            # knowing tool_trace[1] is non-idempotent)
            root_cause_idempotent = scenario["tool_trace"][1].get("idempotent", True)
            if action.decision == "RETRY" and not root_cause_idempotent:
                idempotency_score = 0.99  # Maximum valid score (not 1.0)

            return root_cause_score, idempotency_score

        except Exception:
            # Never let exceptions propagate from graders
            return 0.01, 0.01
