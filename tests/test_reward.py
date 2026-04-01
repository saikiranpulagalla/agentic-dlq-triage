"""Tests for Reward Calculator."""

import sys
sys.path.insert(0, 'src')

import pytest
from dlq_triage.reward import RewardCalculator


def test_reward_max_inputs():
    """Max inputs should score 1.0."""
    reward = RewardCalculator.compute(
        classification_score=1.0,
        transformation_score=1.0,
        root_cause_score=1.0,
        idempotency_score=1.0,
        retry_count=1,
    )
    assert reward.total == 1.0


def test_reward_all_zeros():
    """All zeros should score 0.05 (cost efficiency bonus)."""
    reward = RewardCalculator.compute(
        classification_score=0.0,
        transformation_score=0.0,
        root_cause_score=0.0,
        idempotency_score=0.0,
        retry_count=1,
    )
    assert reward.total == 0.05


def test_reward_high_retry_count_penalty():
    """High retry count (>3) should apply -0.1 penalty."""
    reward = RewardCalculator.compute(
        classification_score=1.0,
        transformation_score=1.0,
        root_cause_score=1.0,
        idempotency_score=1.0,
        retry_count=5,
    )
    # (0.35 + 0.25 + 0.20 + 0.15) - 0.1 = 0.95 - 0.1 = 0.85
    assert abs(reward.total - 0.85) < 0.001


def test_reward_clamping_lower():
    """Negative reward should clamp to 0.0."""
    reward = RewardCalculator.compute(
        classification_score=0.0,
        transformation_score=0.0,
        root_cause_score=0.0,
        idempotency_score=0.0,
        retry_count=10,
    )
    # 0.0 - 0.1 = -0.1, clamped to 0.0
    assert reward.total == 0.0


def test_reward_clamping_upper():
    """Reward > 1.0 should clamp to 1.0."""
    # This shouldn't happen with the formula, but test clamping
    reward = RewardCalculator.compute(
        classification_score=1.0,
        transformation_score=1.0,
        root_cause_score=1.0,
        idempotency_score=1.0,
        retry_count=1,
    )
    assert reward.total <= 1.0


def test_reward_weighted_formula():
    """Test weighted formula calculation."""
    reward = RewardCalculator.compute(
        classification_score=0.9,
        transformation_score=0.8,
        root_cause_score=0.7,
        idempotency_score=0.6,
        retry_count=2,
    )
    # (0.35 * 0.9) + (0.25 * 0.8) + (0.20 * 0.7) + (0.15 * 0.6) + 0.05
    # = 0.315 + 0.20 + 0.14 + 0.09 + 0.05 = 0.795
    assert abs(reward.total - 0.795) < 0.001


def test_reward_breakdown():
    """Test that reward breakdown is correct."""
    reward = RewardCalculator.compute(
        classification_score=0.9,
        transformation_score=0.8,
        root_cause_score=0.7,
        idempotency_score=0.6,
        retry_count=2,
    )
    assert reward.classification_score == 0.9
    assert reward.transformation_score == 0.8
    assert reward.root_cause_score == 0.7
    assert reward.idempotency_score == 0.6
    assert reward.cost_efficiency_score == 0.05


def test_reward_high_retry_penalty():
    """Test penalty for high retry count."""
    reward = RewardCalculator.compute(
        classification_score=0.9,
        transformation_score=0.8,
        root_cause_score=0.7,
        idempotency_score=0.6,
        retry_count=5,
    )
    assert reward.cost_efficiency_score == -0.1
    # (0.35 * 0.9) + (0.25 * 0.8) + (0.20 * 0.7) + (0.15 * 0.6) - 0.1
    # = 0.315 + 0.20 + 0.14 + 0.09 - 0.1 = 0.645
    assert abs(reward.total - 0.645) < 0.001
