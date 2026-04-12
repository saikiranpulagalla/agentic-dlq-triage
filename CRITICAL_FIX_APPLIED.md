# CRITICAL FIX: Cost Efficiency Score Out of Range

## Problem Identified
The evaluator was failing with: "One or more task scores are out of range"

After thorough investigation, the root cause was identified:
- The `cost_efficiency_score` field in the Reward JSON was being set to **-0.1** when retry_count > 3
- The evaluator checks **ALL float fields** in the Reward object, not just the total
- The -0.1 value is outside the required 0-1 range

## Solution Applied
Changed the cost_efficiency_score logic:
- **Before**: `cost_efficiency = -0.1 if retry_count > 3 else 0.05`
- **After**: `cost_efficiency = 0.01 if retry_count > 3 else 0.05`

This ensures ALL float fields in the Reward object are strictly between 0 and 1.

## Files Modified
1. `src/dlq_triage/reward.py` - Updated cost_efficiency calculation
2. `tests/test_reward.py` - Updated test expectations to match new logic

## Verification
✅ All 32 tests passing
✅ All Reward fields now strictly between 0 and 1:
  - classification_score: 0.01-0.99
  - transformation_score: 0.01-0.99
  - root_cause_score: 0.01-0.99
  - idempotency_score: 0.01-0.99
  - cost_efficiency_score: 0.01-0.05 (now always positive)
  - total: 0.01-0.99

## Commits
- GitHub: `77f09fc`
- HF Space: `35f2b4c`

## Why This Matters
The evaluator validates the JSON response from the `/step` endpoint. It checks that every numeric field is strictly between 0 and 1 (exclusive). The cost_efficiency_score was the only field that could violate this constraint.

**Status: READY FOR RESUBMISSION**
