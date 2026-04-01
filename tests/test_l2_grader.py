"""Tests for L2 Grader."""

import sys
sys.path.insert(0, 'src')

import pytest
from dlq_triage.models import Action
from dlq_triage.graders.l2_grader import L2Grader
from dlq_triage.failure_generator import FailureGenerator


@pytest.fixture
def l2_scenario():
    """Get L2 scenario."""
    gen = FailureGenerator(seed=42)
    return gen.generate(2)


def test_l2_correct_transform(l2_scenario):
    """Correct transformation should score 1.0."""
    action = Action(
        decision="TRANSFORM_AND_RETRY",
        transformed_payload={"amount": 150.00, "currency": "USD"},
    )
    score = L2Grader.grade(action, l2_scenario)
    assert score == 1.0


def test_l2_correct_action_wrong_types(l2_scenario):
    """Correct action but wrong types should score 0.4."""
    action = Action(
        decision="TRANSFORM_AND_RETRY",
        transformed_payload={"amount": "150.00", "currency": "USD"},
    )
    score = L2Grader.grade(action, l2_scenario)
    assert score == 0.4


def test_l2_correct_action_no_payload(l2_scenario):
    """Correct action but no payload should score 0.4."""
    action = Action(decision="TRANSFORM_AND_RETRY")
    score = L2Grader.grade(action, l2_scenario)
    assert score == 0.4


def test_l2_wrong_action_retry(l2_scenario):
    """Wrong action (RETRY) should score 0.0."""
    action = Action(decision="RETRY")
    score = L2Grader.grade(action, l2_scenario)
    assert score == 0.0


def test_l2_wrong_action_skip(l2_scenario):
    """Wrong action (SKIP) should score 0.0."""
    action = Action(decision="SKIP")
    score = L2Grader.grade(action, l2_scenario)
    assert score == 0.0


def test_l2_wrong_action_escalate(l2_scenario):
    """Wrong action (ESCALATE) should score 0.0."""
    action = Action(decision="ESCALATE")
    score = L2Grader.grade(action, l2_scenario)
    assert score == 0.0


def test_l2_partial_payload(l2_scenario):
    """Partial payload with correct types should score 0.8."""
    action = Action(
        decision="TRANSFORM_AND_RETRY",
        transformed_payload={"amount": 150.00, "currency": "USD", "extra": "field"},
    )
    score = L2Grader.grade(action, l2_scenario)
    # Should match types (0.4) but not exact (0.2) = 0.8
    assert score == 0.8


def test_l2_grader_never_raises():
    """Grader should never raise exceptions."""
    invalid_scenario = None
    action = Action(decision="TRANSFORM_AND_RETRY")
    
    score = L2Grader.grade(action, invalid_scenario)
    assert score == 0.0
