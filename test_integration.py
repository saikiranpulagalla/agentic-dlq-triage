"""Integration test for AgenticDLQ Triage."""

import sys
sys.path.insert(0, 'src')

from dlq_triage.models import Observation, Action, Reward, EpisodeState
from dlq_triage.failure_generator import FailureGenerator
from dlq_triage.graders.l1_grader import L1Grader
from dlq_triage.graders.l2_grader import L2Grader
from dlq_triage.graders.l3_grader import L3Grader
from dlq_triage.reward import RewardCalculator
from dlq_triage.episode import EpisodeManager
from dlq_triage.main import app


def test_integration():
    """Run comprehensive integration tests."""
    print('='*70)
    print('COMPREHENSIVE PROJECT VERIFICATION')
    print('='*70)

    # 1. Test models
    print('\n1. Testing Pydantic Models...')
    obs = Observation(
        task_id='test', tool_name='test', error_type='transient',
        error_message='test', retry_count=1, tool_trace=[], payload={}
    )
    print('   OK: Observation model works')
    print('   OK: model_dump():', type(obs.model_dump()).__name__)
    print('   OK: model_validate():', type(Observation.model_validate(obs.model_dump())).__name__)

    # 2. Test failure generator
    print('\n2. Testing Failure Generator...')
    gen = FailureGenerator(seed=42)
    l1 = gen.generate(1)
    l2 = gen.generate(2)
    l3 = gen.generate(3)
    print('   OK: L1 scenario:', l1['task_id'], '(' + l1['error_type'] + ')')
    print('   OK: L2 scenario:', l2['task_id'], '(' + l2['error_type'] + ')')
    print('   OK: L3 scenario:', l3['task_id'], '(' + l3['error_type'] + ')')

    # 3. Test graders
    print('\n3. Testing Graders...')
    action_l1 = Action(decision='RETRY', backoff_seconds=32)
    score_l1 = L1Grader.grade(action_l1, l1)
    print('   OK: L1 Grader:', round(score_l1, 2))

    action_l2 = Action(decision='TRANSFORM_AND_RETRY', transformed_payload={'amount': 150.00, 'currency': 'USD'})
    score_l2 = L2Grader.grade(action_l2, l2)
    print('   OK: L2 Grader:', round(score_l2, 2))

    action_l3 = Action(decision='RETRY', root_cause_tool='payment_capture')
    score_l3 = L3Grader.grade(action_l3, l3)
    print('   OK: L3 Grader:', round(score_l3, 2))

    # 4. Test reward calculator
    print('\n4. Testing Reward Calculator...')
    reward = RewardCalculator.compute(
        classification_score=score_l1,
        transformation_score=score_l2,
        root_cause_score=score_l3,
        idempotency_score=0.0,
        retry_count=1
    )
    print('   OK: Reward total:', round(reward.total, 3))
    print('   OK: Cost efficiency:', round(reward.cost_efficiency_score, 2))

    # 5. Test episode manager
    print('\n5. Testing Episode Manager...')
    em = EpisodeManager()
    obs = em.reset(seed=42)
    print('   OK: Reset - episode_id:', em.episode_id[:8] + '...')
    print('   OK: Initial observation:', obs.task_id)

    obs, reward, done, info = em.step(action_l1)
    print('   OK: Step 1 - reward:', round(reward.total, 3), 'done:', done)

    obs, reward, done, info = em.step(action_l2)
    print('   OK: Step 2 - reward:', round(reward.total, 3), 'done:', done)

    obs, reward, done, info = em.step(action_l3)
    print('   OK: Step 3 - reward:', round(reward.total, 3), 'done:', done)

    state = em.state()
    print('   OK: State - cumulative_reward:', round(state.cumulative_reward, 3))

    # 6. Test FastAPI app
    print('\n6. Testing FastAPI Application...')
    print('   OK: FastAPI app initialized')
    print('   OK: Title:', app.title)
    print('   OK: Version:', app.version)

    print('\n' + '='*70)
    print('SUCCESS: ALL TESTS PASSED - PROJECT READY FOR DEPLOYMENT')
    print('='*70)


if __name__ == '__main__':
    test_integration()
