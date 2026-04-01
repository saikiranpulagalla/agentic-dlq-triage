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
        - root_cause_score: +0.3 if root cause correctly identified, +0.4 bonus for perfect answer
        - idempotency_score: +0.3 if agent correctly chose RETRY knowing tool is non-idempotent
        
        Args:
            action: The agent's action
            scenario: The failure scenario
            
        Returns:
            Tuple of (root_cause_score, idempotency_score)
        """
        try:
            root_cause_score = 0.0
            idempotency_score = 0.0

            # root_cause_score (+0.3 if root cause correctly identified)
            # maps to whether action.root_cause_tool matches scenario root
            if action.root_cause_tool == scenario.get("root_cause_tool"):
                root_cause_score += 1.0  # Full score for correct identification

            # idempotency_score (+0.3 if agent correctly chose RETRY
            # knowing tool_trace[1] is non-idempotent)
            root_cause_idempotent = scenario["tool_trace"][1].get("idempotent", True)
            if action.decision == "RETRY" and not root_cause_idempotent:
                idempotency_score += 1.0  # Full score for correct idempotency handling

            # No additional bonus needed - the weights handle the scoring

            return root_cause_score, idempotency_score

        except Exception:
            # Never let exceptions propagate from graders
            return 0.0, 0.0
