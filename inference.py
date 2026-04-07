"""AgenticDLQ Triage — Baseline inference script

Runs two agents against the environment and prints a comparison table.
"""

import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables - required for submission
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default - required
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://router.huggingface.co/v1")

BASE_URL = API_BASE_URL


# ── Rule-based agent ──────────────────────────────────────────────────────────
def rule_based_action(observation: dict) -> dict:
    """Apply rule-based policy to observation."""
    error_type = observation.get("error_type", "")
    retry_count = observation.get("retry_count", 0)

    if error_type == "transient":
        return {
            "decision": "RETRY",
            "backoff_seconds": 32,
            "transformed_payload": None,
            "root_cause_tool": None,
        }

    elif error_type == "schema_mismatch":
        payload = observation.get("payload", {})
        # Fix known type issue: str amount -> float
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
        # Find first non-idempotent failed tool
        trace = observation.get("tool_trace", [])
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


TASK_IDS = ["task_l1", "task_l2", "task_l3"]


def run_episode(agent_fn, seed: int = 42) -> list[float]:
    """Run one full episode and return per-task scores."""
    scores = []
    resp = requests.post(f"{BASE_URL}/reset", json={"seed": seed})
    resp.raise_for_status()
    data = resp.json()
    obs = data["observation"]
    done = False
    step_num = 0

    while not done:
        task_id = TASK_IDS[step_num] if step_num < len(TASK_IDS) else f"task_{step_num}"
        action = agent_fn(obs)
        action_str = action.get("decision", "UNKNOWN")

        print(f"[START] task={task_id} env=agentic-dlq-triage model={MODEL_NAME}", flush=True)

        step_resp = requests.post(f"{BASE_URL}/step", json=action)
        step_resp.raise_for_status()
        result = step_resp.json()
        raw_reward = float(result["reward"]["total"])
        # Cap the reward as suggested in Discord
        reward = max(0.05, min(0.95, raw_reward))
        reward = round(reward, 2)
        done = result["done"]
        obs = result["observation"]
        scores.append(reward)

        print(f"[STEP] action={action_str}", flush=True)
        print(f"[END] task={task_id} score={reward:.2f} steps=1", flush=True)

        step_num += 1

    return scores


def run_rule_based(seed: int = 42) -> list:
    """Run rule-based agent and return scores."""
    return run_episode(rule_based_action, seed)


# ── LLM agent (Groq via OpenAI-compatible client) ────────────────────────────
def build_llm_prompt(observation: dict) -> str:
    """Build prompt for LLM agent."""
    error_type = observation.get("error_type", "")
    
    base = f"""You are an expert SRE diagnosing a Dead Letter Queue (DLQ) failure. Analyze this failed tool call carefully and decide the best recovery action.

Observation:
{json.dumps(observation, indent=2)}

DECISION RULES:
- error_type "transient": choose RETRY. Set backoff_seconds to the number 
  of seconds in retry_after from the error_message (e.g. "retry_after: 32s" → 32).

- error_type "schema_mismatch": choose TRANSFORM_AND_RETRY. 
  Fix the payload type errors shown in error_message.
  Put the corrected payload in transformed_payload.

- error_type "cascading": Carefully read tool_trace step by step.
  Find the FIRST tool that actually failed (not a downstream victim).
  A tool with status "timeout" or "failed" that has a non-null error is the root cause.
  A tool that failed because it was missing input from a previous tool is a VICTIM, not the cause.
  Set root_cause_tool to the NAME of the root cause tool (the "tool" field in that trace entry).
  Choose RETRY.

- error_type "silent_corruption": The tool returned success but output is corrupt.
  Look for null values, empty strings, or UNKNOWN status in the tool output.
  Choose ESCALATE.

Reply with ONLY valid JSON. No explanation. No markdown fences:
{{
  "decision": "RETRY" | "SKIP" | "ESCALATE" | "TRANSFORM_AND_RETRY",
  "backoff_seconds": <integer or null>,
  "transformed_payload": <dict or null>,
  "root_cause_tool": <exact tool name string from trace, or null>
}}"""
    return base


def run_llm_agent(seed: int = 42) -> list[float]:
    """Run LLM agent and return scores."""
    client = OpenAI(base_url=LLM_API_BASE_URL, api_key=HF_TOKEN)

    def llm_action(obs: dict) -> dict:
        prompt = build_llm_prompt(obs)
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
            print(f"[STEP] action=ESCALATE", flush=True)
            return {
                "decision": "ESCALATE",
                "transformed_payload": None,
                "root_cause_tool": None,
                "backoff_seconds": None,
            }

    return run_episode(llm_action, seed)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Run both agents and print comparison table."""
    model_label = MODEL_NAME.split("/")[-1][:20]

    rule_scores = run_rule_based(seed=42)
    llm_scores = run_llm_agent(seed=42)

    print("\n" + "=" * 52)
    print(f"{'Task':<26} {'Rule-Based':>10} {f'LLM ({model_label})':>12}")
    print("-" * 52)
    task_names = ["L1 Transient", "L2 Schema", "L3 Cascade"]
    for name, rb, llm in zip(task_names, rule_scores, llm_scores):
        print(f"{name:<26} {rb:>10.2f} {llm:>12.2f}")
    print("=" * 52)
    print(f"{'Average':<26} {sum(rule_scores)/len(rule_scores):>10.2f} {sum(llm_scores)/len(llm_scores):>12.2f}")


if __name__ == "__main__":
    main()
