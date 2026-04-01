"""Reward calculation for AgenticDLQ Triage environment."""

from dlq_triage.models import Reward


class RewardCalculator:
    """Computes reward based on multiple scoring components."""

    @staticmethod
    def compute(
        classification_score: float,
        transformation_score: float,
        root_cause_score: float,
        idempotency_score: float,
        retry_count: int,
    ) -> Reward:
        """Compute total reward from component scores.
        
        Formula:
        - cost_efficiency = -0.1 if retry_count > 3 else 0.05
        - reward = (0.35 * classification_score +
                    0.25 * transformation_score +
                    0.20 * root_cause_score +
                    0.15 * idempotency_score +
                    cost_efficiency)
        - total = clamp(reward, 0.0, 1.0)
        
        Args:
            classification_score: Score for error classification (0.0-1.0)
            transformation_score: Score for payload transformation (0.0-1.0)
            root_cause_score: Score for root cause identification (0.0-1.0)
            idempotency_score: Score for idempotency handling (0.0-1.0)
            retry_count: Number of retries attempted
            
        Returns:
            Reward object with breakdown and total
        """
        # Calculate cost efficiency penalty/bonus
        cost_efficiency = -0.1 if retry_count > 3 else 0.05

        # Calculate weighted reward
        reward = (
            0.35 * classification_score
            + 0.25 * transformation_score
            + 0.20 * root_cause_score
            + 0.15 * idempotency_score
            + cost_efficiency
        )

        # Clamp to [0.0, 1.0]
        total = max(0.0, min(1.0, reward))

        return Reward(
            classification_score=classification_score,
            transformation_score=transformation_score,
            root_cause_score=root_cause_score,
            idempotency_score=idempotency_score,
            cost_efficiency_score=cost_efficiency,
            total=total,
        )
