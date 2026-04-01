"""Deterministic failure scenario generator for AgenticDLQ Triage."""

import random
from typing import Dict, Any


class FailureGenerator:
    """Generates deterministic failure scenarios based on task level and seed."""

    def __init__(self, seed: int):
        """Initialize with a seed for reproducible randomness.
        
        Args:
            seed: Random seed for reproducibility (uses random.Random, not global state)
        """
        self.rng = random.Random(seed)
        self.seed = seed

    def generate(self, task_level: int) -> Dict[str, Any]:
        """Generate a failure scenario for the given task level.
        
        Args:
            task_level: 1 (transient), 2 (schema_mismatch), or 3 (cascading)
            
        Returns:
            Dictionary containing the failure scenario
        """
        if task_level == 1:
            return self._generate_l1_transient()
        elif task_level == 2:
            return self._generate_l2_schema_mismatch()
        elif task_level == 3:
            return self._generate_l3_cascading()
        else:
            raise ValueError(f"Invalid task_level: {task_level}. Must be 1, 2, or 3.")

    def _generate_l1_transient(self) -> Dict[str, Any]:
        """Generate L1 transient failure scenario.
        
        Target score: ~0.92
        Correct action: RETRY with backoff_seconds=32
        """
        return {
            "task_id": "dlq_001",
            "tool_name": "stripe_charge_api",
            "error_type": "transient",
            "error_message": (
                "HTTPError 429: Too Many Requests — tool: stripe_charge_api, "
                "retry_after: 32s, request_id: req_8fGh2k"
            ),
            "retry_count": 1,
            "retry_after_seconds": 32,
            "idempotency_key": "stripe_req_8fGh2k_v1",
            "tool_trace": [
                {
                    "call_id": 1,
                    "tool": "stripe_charge_api",
                    "status": "failed",
                    "error": "rate_limit",
                    "timestamp": "2026-03-31T10:00:01Z",
                }
            ],
            "payload": {"amount": 150.00, "currency": "USD", "customer_id": "cus_abc123"},
            "correct_action": "RETRY",
        }

    def _generate_l2_schema_mismatch(self) -> Dict[str, Any]:
        """Generate L2 schema mismatch failure scenario.
        
        Target score: ~0.78
        Correct action: TRANSFORM_AND_RETRY with corrected payload
        """
        return {
            "task_id": "dlq_002",
            "tool_name": "payment_processor",
            "error_type": "schema_mismatch",
            "error_message": (
                "ValidationError: field 'amount' expected float, got str. "
                "Received: {'amount': '150.00'}. Schema expects: {'amount': float}"
            ),
            "retry_count": 2,
            "idempotency_key": None,
            "tool_trace": [
                {
                    "call_id": 1,
                    "tool": "payment_processor",
                    "status": "failed",
                    "error": "schema_mismatch",
                },
                {
                    "call_id": 2,
                    "tool": "payment_processor",
                    "status": "failed",
                    "error": "schema_mismatch",
                },
            ],
            "payload": {"amount": "150.00", "currency": "USD"},
            "expected_payload": {"amount": 150.00, "currency": "USD"},
            "correct_action": "TRANSFORM_AND_RETRY",
        }

    def _generate_l3_cascading(self) -> Dict[str, Any]:
        """Generate L3 cascading failure scenario.
        
        Target score: ~0.65
        Correct action: RETRY with root_cause_tool="payment_capture"
        """
        return {
            "task_id": "dlq_003",
            "tool_name": "order_fulfillment_pipeline",
            "error_type": "cascading",
            "error_message": (
                "Pipeline failure: tool_1 succeeded, tool_2 timed out after 30s, "
                "tool_3 failed due to missing input from tool_2"
            ),
            "retry_count": 1,
            "idempotency_key": "ord_9871_payment_v1",
            "root_cause_tool": "payment_capture",
            "tool_trace": [
                {
                    "call_id": 1,
                    "tool": "inventory_check",
                    "status": "success",
                    "idempotent": True,
                    "output": {"item_id": "SKU_001", "available": True},
                },
                {
                    "call_id": 2,
                    "tool": "payment_capture",
                    "status": "timeout",
                    "error": "timeout_30s",
                    "idempotent": False,
                    "output": None,
                },
                {
                    "call_id": 3,
                    "tool": "shipping_label",
                    "status": "failed",
                    "error": "missing_input: payment_id",
                    "idempotent": True,
                    "output": None,
                },
            ],
            "payload": {"order_id": "ORD_9871", "customer_id": "cus_xyz456"},
            "correct_action": "RETRY",
        }
