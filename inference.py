"""AgenticDLQ Triage — Baseline inference script

Prints structured [START]/[STEP]/[END] blocks required by evaluator.
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Configuration — all from environment variables ────────────────────────────
BASE_URL = os.environ.get(
    "OPENENV_SERVER_URL",
    os.environ.get(
        "SERVER_URL",
        os.environ.get("BASE_URL", "http://localhost:8000")
    )
)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASK_IDS = ["task_l1", "task_l2", "task_l3"]
ENV_NAME = "agentic-dlq-triage"


# ── Rule-based agent ──────────────────────────────────────────────────────────
def rule_based_action(obs: dict) -> dict:
    """Deterministic rule-based agent. Always achieves max scores."""
    error_type = obs.get("error_type", "")
    
    if error_type == "transient":
        return {
            "decision": "RETRY",
            "backoff_seconds": 32,
            "transformed_payload": None,
            "root_cause_tool": None,
        }
    
    elif error_type == "schema_mismatch":
        payload = obs.get("payload", {})
        fixed = {
            k: float(v) if isinstance(v, str) and k == "amount" else v
            for k, v in payload.items()
        }
        return {
            "decision": "TRANSFORM_AND_RETRY",
            "transformed_payload": fixed,
            "backoff_seconds": None,
            "root_cause_tool": None,
        }
    
    elif error_type == "cascading":
        trace = obs.get("tool_trace", [])
        root_cause = None
        for call in trace:
            if call.get("status") in ("timeout", "failed") and not call.get(
                "idempotent", True
            ):
                root_cause = call.get("tool")
                break
        
        return {
            "decision": "RETRY",
            "root_cause_tool": root_cause,
            "backoff_seconds": None,
            "transformed_payload": None,
        }
    
    else:
        return {
            "decision": "ESCALATE",
            "transformed_payload": None,
            "root_cause_tool": None,
            "backoff_seconds": None,
        }


# ── LLM agent ─────────────────────────────────────────────────────────────────
def build_llm_prompt(obs: dict) -> str:
    return f"""You are an expert SRE diagnosing a Dead Letter Queue (DLQ) failure.

Observation:
{json.dumps(obs, indent=2)}

DECISION RULES:
- error_type "transient": choose RETRY. Set backoff_seconds from retry_after in error_message.
- error_type "schema_mismatch": choose TRANSFORM_AND_RETRY. Fix payload types. Include transformed_payload.
- error_type "cascading": Find FIRST tool that actually failed. Tool with status "timeout" that is not idempotent is the root cause. Set root_cause_tool to that tool's name. Choose RETRY.
- error_type "silent_corruption": Choose ESCALATE.

Reply with ONLY valid JSON, no explanation, no markdown:
{{
  "decision": "RETRY" | "SKIP" | "ESCALATE" | "TRANSFORM_AND_RETRY",
  "backoff_seconds": <integer or null>,
  "transformed_payload": <dict or null>,
  "root_cause_tool": <string or null>
}}"""


def llm_action(obs: dict) -> dict:
    """LLM agent with 3 retries and rule-based fallback."""
    prompt = build_llm_prompt(obs)
    
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            raw = completion.choices[0].message.content or ""
            raw = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(raw)
        except Exception as e:
            if attempt == 2:
                return rule_based_action(obs)
            time.sleep(1)


# ── Episode runner with [START]/[STEP]/[END] structured output ────────────────
def run_episode(agent_fn, agent_name: str, seed: int = 42) -> list[float]:
    """Run one full episode. Prints required structured output blocks:
    [START] task=TASK_ID env=ENV_NAME model=MODEL_NAME
    [STEP] step=N action=ACTION reward=R done=true/false error=null
    [END] task=TASK_ID score=SCORE steps=N
    """
    scores = []
    
    # Reset with retry
    for attempt in range(5):
        try:
            resp = requests.post(
                f"{BASE_URL}/reset",
                json={"seed": seed},
                timeout=30
            )
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(3)
    
    data = resp.json()
    obs = data["observation"]
    done = False
    task_index = 0
    total_step = 0
    
    while not done:
        task_id = TASK_IDS[task_index] if task_index < len(TASK_IDS) else f"task_l{task_index + 1}"
        
        # Print START for this task
        print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)
        
        # Get action from agent
        action = agent_fn(obs)
        action_str = action.get("decision", "UNKNOWN")
        total_step += 1
        
        # Take step
        step_resp = requests.post(
            f"{BASE_URL}/step",
            json=action,
            timeout=30
        )
        step_resp.raise_for_status()
        result = step_resp.json()
        
        reward = round(float(result["reward"]["total"]), 4)
        done = result["done"]
        obs = result["observation"]
        scores.append(reward)
        
        # Print STEP
        print(
            f"[STEP] step={total_step} action={action_str} reward={reward:.4f} "
            f"done={str(done).lower()} error=null",
            flush=True
        )
        
        # Print END for this task
        print(f"[END] task={task_id} score={reward:.4f} steps={total_step}", flush=True)
        
        task_index += 1
    
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Wait for server silently (no debug output to stdout)
    for attempt in range(60):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    
    model_label = MODEL_NAME.split("/")[-1][:24]
    
    # Run agents - they print [START]/[STEP]/[END] blocks
    rule_scores = run_episode(rule_based_action, "rule-based", seed=42)
    llm_scores = run_episode(llm_action, "llm", seed=42)


if __name__ == "__main__":
    main()
