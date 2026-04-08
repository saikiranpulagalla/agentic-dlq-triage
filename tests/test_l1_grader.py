"""Tests for L1 Grader."""

import sys
sys.path.insert(0, 'src')

import pytest
from dlq_triage.models import Action
from dlq_triage.graders.l1_grader import L1Grader
from dlq_triage.failure_generator import FailureGenerator


@pytest.fixture
def l1_scenario():
    """Get L1 scenario."""
    gen = FailureGenerator(seed=42)
    return gen.generate(1)


def test_l1_retry_with_correct_backoff(l1_scenario):
    """RETRY with correct backoff should score 0.99."""
    action = Action(decision="RETRY", backoff_seconds=32)
    score = L1Grader.grade(action, l1_scenario)
    assert score == 0.99


def test_l1_retry_without_backoff(l1_scenario):
    """RETRY without backoff should score 0.6."""
    action = Action(decision="RETRY")
    score = L1Grader.grade(action, l1_scenario)
    assert score == 0.6


def test_l1_retry_with_close_backoff(l1_scenario):
    """RETRY with backoff within 10s should score 0.99."""
    action = Action(decision="RETRY", backoff_seconds=35)  # 32 + 3
    score = L1Grader.grade(action, l1_scenario)
    assert score == 0.99


def test_l1_retry_with_far_backoff(l1_scenario):
    """RETRY with backoff > 10s away should score 0.6."""
    action = Action(decision="RETRY", backoff_seconds=50)  # 32 + 18
    score = L1Grader.grade(action, l1_scenario)
    assert score == 0.6


def test_l1_skip_decision(l1_scenario):
    """SKIP decision should score 0.01."""
    action = Action(decision="SKIP")
    score = L1Grader.grade(action, l1_scenario)
    assert score == 0.01


def test_l1_escalate_decision(l1_scenario):
    """ESCALATE decision should score 0.01."""
    action = Action(decision="ESCALATE")
    score = L1Grader.grade(action, l1_scenario)
    assert score == 0.01


def test_l1_transform_decision(l1_scenario):
    """TRANSFORM_AND_RETRY decision should score 0.01."""
    action = Action(decision="TRANSFORM_AND_RETRY")
    score = L1Grader.grade(action, l1_scenario)
    assert score == 0.01


def test_l1_grader_never_raises():
    """Grader should never raise exceptions."""
    # Create invalid scenario (missing retry_after_seconds)
    invalid_scenario = {"retry_after_seconds": None}
    action = Action(decision="RETRY")
    
    # Should not raise, should return 0.6 (RETRY without backoff)
    score = L1Grader.grade(action, invalid_scenario)
    assert score == 0.6
