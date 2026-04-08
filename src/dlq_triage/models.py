"""Pydantic models for AgenticDLQ Triage environment."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


class Observation(BaseModel):
    """Observation returned by the environment after reset or step."""

    task_id: str = Field(..., description="Unique identifier for the task")
    tool_name: str = Field(..., description="Name of the tool that failed")
    error_type: Literal["transient", "schema_mismatch", "cascading", "silent_corruption"] = Field(
        ..., description="Classification of the error type"
    )
    error_message: str = Field(..., description="Detailed error message")
    retry_count: int = Field(..., description="Number of retries attempted so far")
    tool_trace: List[Dict[str, Any]] = Field(
        default_factory=list, description="Execution trace of tool calls"
    )
    payload: Dict[str, Any] = Field(..., description="Current payload being processed")
    idempotency_key: Optional[str] = Field(
        default=None, description="Key for idempotent operations"
    )


class Action(BaseModel):
    """Action taken by the agent."""

    decision: Literal["RETRY", "SKIP", "ESCALATE", "TRANSFORM_AND_RETRY"] = Field(
        ..., description="Decision on how to handle the failure"
    )
    transformed_payload: Optional[Dict[str, Any]] = Field(
        default=None, description="Transformed payload for TRANSFORM_AND_RETRY"
    )
    root_cause_tool: Optional[str] = Field(
        default=None, description="Identified root cause tool for cascading failures"
    )
    backoff_seconds: Optional[int] = Field(
        default=None, description="Backoff time in seconds for retry"
    )


class Reward(BaseModel):
    """Reward breakdown for an action."""

    classification_score: float = Field(..., description="Score for correct error classification (0.01-0.99)")
    transformation_score: float = Field(..., description="Score for payload transformation (0.01-0.99)")
    root_cause_score: float = Field(..., description="Score for root cause identification (0.01-0.99)")
    idempotency_score: float = Field(..., description="Score for idempotency handling (0.01-0.99)")
    cost_efficiency_score: float = Field(..., description="Score for cost efficiency")
    total: float = Field(..., description="Total reward (strictly between 0.01 and 0.99)")


class EpisodeState(BaseModel):
    """Current state of an episode."""

    episode_id: str = Field(..., description="Unique episode identifier")
    task_level: int = Field(..., description="Current task level (1, 2, or 3)")
    current_step: int = Field(..., description="Current step count")
    is_done: bool = Field(..., description="Whether episode is complete")
    cumulative_reward: float = Field(..., description="Sum of all rewards so far")
    last_action: Optional[Action] = Field(default=None, description="Last action taken")
    last_reward: Optional[Reward] = Field(default=None, description="Last reward received")
    seed: int = Field(..., description="Seed used for reproducibility")
