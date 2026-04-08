"""L1 Grader: Transient failure recovery scoring."""

from typing import Dict, Any
from dlq_triage.models import Action


class L1Grader:
    """Grades agent actions for L1 transient failure scenarios."""

    @staticmethod
    def grade(action: Action, scenario: Dict[str, Any]) -> float:
        """Grade the agent's action for a transient failure.
        
        Scoring logic:
        - If decision != "RETRY": return 0.01 (minimum valid score)
        - If decision == "RETRY": score = 0.6
        - If backoff_seconds is set AND abs(backoff - retry_after) <= 10: score += 0.39
        
        Args:
            action: The agent's action
            scenario: The failure scenario
            
        Returns:
            Score strictly between 0.0 and 1.0
        """
        try:
            # Check if decision is RETRY
            if action.decision != "RETRY":
                return 0.01  # Minimum valid score (not 0.0)

            score = 0.6

            # Check backoff accuracy
            if action.backoff_seconds is not None:
                retry_after = scenario.get("retry_after_seconds", 32)
                if abs(action.backoff_seconds - retry_after) <= 10:
                    score += 0.39  # Total becomes 0.99 (not 1.0)

            return score

        except Exception:
            # Never let exceptions propagate from graders
            return 0.01
