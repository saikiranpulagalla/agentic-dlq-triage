"""Tests for Reward Calculator."""

import sys
sys.path.insert(0, 'src')

import pytest
from dlq_triage.reward import RewardCalculator


def test_reward_max_inputs():
    """Max inputs should score 0.99 (clamped from 1.0)."""
    reward = RewardCalculator.compute(
        classification_score=0.99,
        transformation_score=0.99,
        root_cause_score=0.99,
        idempotency_score=0.99,
        retry_count=1,
    )
    # (0.35 + 0.25 + 0.20 + 0.15) * 0.99 + 0.05 = 0.95 * 0.99 + 0.05 = 0.9405 + 0.05 = 0.9905, clamped to 0.99
    assert reward.total == 0.99


def test_reward_all_zeros():
    """All zeros should score 0.0595 (cost efficiency bonus)."""
    reward = RewardCalculator.compute(
        classification_score=0.01,
        transformation_score=0.01,
        root_cause_score=0.01,
        idempotency_score=0.01,
        retry_count=1,
    )
    # (0.35 + 0.25 + 0.20 + 0.15) * 0.01 + 0.05 = 0.95 * 0.01 + 0.05 = 0.0095 + 0.05 = 0.0595
    assert abs(reward.total - 0.0595) < 0.001


def test_reward_high_retry_count_penalty():
    """High retry count (>3) should apply -0.1 penalty."""
    reward = RewardCalculator.compute(
        classification_score=0.99,
        transformation_score=0.99,
        root_cause_score=0.99,
        idempotency_score=0.99,
        retry_count=5,
    )
    # (0.35 + 0.25 + 0.20 + 0.15) * 0.99 - 0.1 = 0.95 * 0.99 - 0.1 = 0.9405 - 0.1 = 0.8405
    assert abs(reward.total - 0.8405) < 0.001


def test_reward_clamping_lower():
    """Negative reward should clamp to 0.01."""
    reward = RewardCalculator.compute(
        classification_score=0.01,
        transformation_score=0.01,
        root_cause_score=0.01,
        idempotency_score=0.01,
        retry_count=10,
    )
    # (0.35 + 0.25 + 0.20 + 0.15) * 0.01 - 0.1 = 0.95 * 0.01 - 0.1 = 0.0095 - 0.1 = -0.0905, clamped to 0.01
    assert reward.total == 0.01


def test_reward_clamping_upper():
    """Reward > 0.99 should clamp to 0.99."""
    reward = RewardCalculator.compute(
        classification_score=0.99,
        transformation_score=0.99,
        root_cause_score=0.99,
        idempotency_score=0.99,
        retry_count=1,
    )
    assert reward.total <= 0.99


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
