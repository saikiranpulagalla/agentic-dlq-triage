"""Tests for L3 Grader."""

import sys
sys.path.insert(0, 'src')

import pytest
from dlq_triage.models import Action
from dlq_triage.graders.l3_grader import L3Grader
from dlq_triage.failure_generator import FailureGenerator


@pytest.fixture
def l3_scenario():
    """Get L3 scenario."""
    gen = FailureGenerator(seed=42)
    return gen.generate(3)


def test_l3_all_correct(l3_scenario):
    """All correct should score 1.98 total (0.99 + 0.99)."""
    action = Action(decision="RETRY", root_cause_tool="payment_capture")
    root_cause, idempotency = L3Grader.grade(action, l3_scenario)
    assert root_cause + idempotency == 1.98


def test_l3_wrong_root_cause(l3_scenario):
    """Wrong root cause should score 0.99 (from RETRY + non-idempotent check)."""
    action = Action(decision="RETRY", root_cause_tool="inventory_check")
    root_cause, idempotency = L3Grader.grade(action, l3_scenario)
    assert root_cause + idempotency == 1.0  # 0.01 + 0.99


def test_l3_right_cause_wrong_decision(l3_scenario):
    """Right cause but wrong decision should score 1.0."""
    action = Action(decision="SKIP", root_cause_tool="payment_capture")
    root_cause, idempotency = L3Grader.grade(action, l3_scenario)
    assert root_cause + idempotency == 1.0  # 0.99 + 0.01


def test_l3_no_root_cause_retry(l3_scenario):
    """RETRY without root cause should score 1.0 (from non-idempotent check)."""
    action = Action(decision="RETRY")
    root_cause, idempotency = L3Grader.grade(action, l3_scenario)
    assert root_cause + idempotency == 1.0  # 0.01 + 0.99


def test_l3_wrong_cause_wrong_decision(l3_scenario):
    """Wrong cause and wrong decision should score 0.02."""
    action = Action(decision="SKIP", root_cause_tool="inventory_check")
    root_cause, idempotency = L3Grader.grade(action, l3_scenario)
    assert root_cause + idempotency == 0.02  # 0.01 + 0.01


def test_l3_escalate_decision(l3_scenario):
    """ESCALATE decision should score 1.0 (from root cause match only)."""
    action = Action(decision="ESCALATE", root_cause_tool="payment_capture")
    root_cause, idempotency = L3Grader.grade(action, l3_scenario)
    # ESCALATE doesn't match RETRY, so no 0.99 from non-idempotent check
    # But root_cause matches, so 0.99 + 0.01
    assert root_cause + idempotency == 1.0


def test_l3_transform_decision(l3_scenario):
    """TRANSFORM_AND_RETRY decision should score 1.0 (from root cause match)."""
    action = Action(decision="TRANSFORM_AND_RETRY", root_cause_tool="payment_capture")
    root_cause, idempotency = L3Grader.grade(action, l3_scenario)
    # Root cause matches (0.99), but decision is not RETRY so no additional points (0.01)
    assert root_cause + idempotency == 1.0


def test_l3_grader_never_raises():
    """Grader should never raise exceptions."""
    invalid_scenario = None
    action = Action(decision="RETRY", root_cause_tool="payment_capture")
    
    root_cause, idempotency = L3Grader.grade(action, invalid_scenario)
    assert root_cause + idempotency == 0.02  # 0.01 + 0.01
