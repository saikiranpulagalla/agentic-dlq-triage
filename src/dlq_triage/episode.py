"""Episode management for AgenticDLQ Triage environment."""

import uuid
from dlq_triage.models import Observation, Action, Reward, EpisodeState
from dlq_triage.failure_generator import FailureGenerator
from dlq_triage.graders.l1_grader import L1Grader
from dlq_triage.graders.l2_grader import L2Grader
from dlq_triage.graders.l3_grader import L3Grader
from dlq_triage.reward import RewardCalculator


class EpisodeManager:
    """Manages episode state and transitions."""

    MAX_STEPS = 100

    def __init__(self):
        """Initialize episode manager."""
        self.episode_id = str(uuid.uuid4())
        self.task_level = 1
        self.current_step = 0
        self.is_done = False
        self.cumulative_reward = 0.0
        self.last_action: Action | None = None
        self.last_reward: Reward | None = None
        self.seed = 0
        self.failure_generator: FailureGenerator | None = None
        self.current_scenario: dict | None = None

    def reset(self, seed: int = 42) -> Observation:
        """Reset episode to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
        """
        self.episode_id = str(uuid.uuid4())
        self.task_level = 1
        self.current_step = 0
        self.is_done = False
        self.cumulative_reward = 0.0
        self.last_action = None
        self.last_reward = None
        self.seed = seed
        self.failure_generator = FailureGenerator(seed)
        
        # Generate first scenario
        self.current_scenario = self.failure_generator.generate(self.task_level)
        
        return self._scenario_to_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Execute one step of the episode.
        
        Args:
            action: The agent's action
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1
        self.last_action = action
        
        # Grade the action based on current task level
        if self.task_level == 1:
            classification_score = L1Grader.grade(action, self.current_scenario)
            transformation_score = 0.01  # Not applicable, use minimum valid score
            root_cause_score = 0.01      # Not applicable, use minimum valid score
            idempotency_score = 0.01     # Not applicable, use minimum valid score
        elif self.task_level == 2:
            classification_score = 0.01  # Not applicable, use minimum valid score
            transformation_score = L2Grader.grade(action, self.current_scenario)
            root_cause_score = 0.01      # Not applicable, use minimum valid score
            idempotency_score = 0.01     # Not applicable, use minimum valid score
        elif self.task_level == 3:
            classification_score = 0.01  # Not applicable, use minimum valid score
            transformation_score = 0.01  # Not applicable, use minimum valid score
            root_cause_score, idempotency_score = L3Grader.grade(action, self.current_scenario)
        else:
            classification_score = 0.01
            transformation_score = 0.01
            root_cause_score = 0.01
            idempotency_score = 0.01
        
        # Compute reward
        retry_count = self.current_scenario.get("retry_count", 0)
        reward = RewardCalculator.compute(
            classification_score=classification_score,
            transformation_score=transformation_score,
            root_cause_score=root_cause_score,
            idempotency_score=idempotency_score,
            retry_count=retry_count,
        )
        
        self.last_reward = reward
        self.cumulative_reward += reward.total
        
        # Advance to next task level or mark done
        self.task_level += 1
        if self.task_level > 3 or self.current_step >= self.MAX_STEPS:
            self.is_done = True
        else:
            # Generate next scenario
            self.current_scenario = self.failure_generator.generate(self.task_level)
        
        # Get next observation
        observation = self._scenario_to_observation()
        
        info = {
            "task_level": self.task_level - 1,  # Previous level
            "step": self.current_step,
            "cumulative_reward": self.cumulative_reward,
        }
        
        return observation, reward, self.is_done, info

    def state(self) -> EpisodeState:
        """Get current episode state.
        
        Returns:
            Current episode state
        """
        return EpisodeState(
            episode_id=self.episode_id,
            task_level=self.task_level,
            current_step=self.current_step,
            is_done=self.is_done,
            cumulative_reward=self.cumulative_reward,
            last_action=self.last_action,
            last_reward=self.last_reward,
            seed=self.seed,
        )

    def _scenario_to_observation(self) -> Observation:
        """Convert current scenario to observation.
        
        Returns:
            Observation object
        """
        scenario = self.current_scenario or {}
        
        return Observation(
            task_id=scenario.get("task_id", "unknown"),
            tool_name=scenario.get("tool_name", "unknown"),
            error_type=scenario.get("error_type", "transient"),
            error_message=scenario.get("error_message", ""),
            retry_count=scenario.get("retry_count", 0),
            tool_trace=scenario.get("tool_trace", []),
            payload=scenario.get("payload", {}),
            idempotency_key=scenario.get("idempotency_key"),
        )
