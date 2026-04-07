"""L2 Grader: Schema mismatch transformation scoring."""

from typing import Dict, Any
from dlq_triage.models import Action


class L2Grader:
    """Grades agent actions for L2 schema mismatch scenarios."""

    @staticmethod
    def grade(action: Action, scenario: Dict[str, Any]) -> float:
        """Grade the agent's action for a schema mismatch failure.
        
        Scoring logic:
        - If decision != "TRANSFORM_AND_RETRY": return 0.0
        - score = 0.4
        - If transformed_payload types match expected_payload types: score += 0.4
        - If transformed_payload == expected_payload exactly: score += 0.2
        
        Args:
            action: The agent's action
            scenario: The failure scenario
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            # Check if decision is TRANSFORM_AND_RETRY
            if action.decision != "TRANSFORM_AND_RETRY":
                return 0.0

            score = 0.4

            # Get expected payload
            expected_payload = scenario.get("expected_payload")
            if expected_payload is None:
                return score

            # Check if transformed_payload exists
            if action.transformed_payload is None:
                return score

            # Check if types match
            if _types_match(action.transformed_payload, expected_payload):
                score += 0.4

            # Check if all expected keys match exactly (extra keys are fine)
            if all(
                action.transformed_payload.get(k) == v
                for k, v in expected_payload.items()
            ):
                score += 0.2

            return score

        except Exception:
            # Never let exceptions propagate from graders
            return 0.0


def _types_match(payload1: Dict[str, Any], payload2: Dict[str, Any]) -> bool:
    """Check if two payloads have matching types for all keys.
    
    Args:
        payload1: First payload
        payload2: Second payload
        
    Returns:
        True if all keys in payload2 exist in payload1 with matching types
    """
    try:
        for key, value in payload2.items():
            if key not in payload1:
                return False
            if type(payload1[key]) != type(value):
                return False
        return True
    except Exception:
        return False
