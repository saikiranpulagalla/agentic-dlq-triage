"""FastAPI application for AgenticDLQ Triage environment."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from dlq_triage.models import Observation, Action, Reward, EpisodeState
from dlq_triage.episode import EpisodeManager
from dlq_triage.graders.l1_grader import L1Grader
from dlq_triage.graders.l2_grader import L2Grader
from dlq_triage.graders.l3_grader import L3Grader
from dlq_triage.failure_generator import FailureGenerator

app = FastAPI(
    title="AgenticDLQ Triage",
    description="OpenEnv environment for diagnosing tool-call failures",
    version="1.0.0",
)

# Global episode manager
episode_manager = EpisodeManager()


@app.get("/tasks")
async def get_tasks() -> dict:
    """Get list of available tasks with graders.
    
    Returns:
        Dictionary with tasks information
    """
    return {
        "tasks": [
            {
                "task_id": "task_l1",
                "name": "Transient failure recovery",
                "difficulty": "easy",
                "description": "Agent receives a rate-limit failure trace and must decide to RETRY with correct backoff.",
                "grader": True,
                "score_range": [0.01, 0.99]
            },
            {
                "task_id": "task_l2", 
                "name": "Schema mismatch transform",
                "difficulty": "medium",
                "description": "Agent must identify type mismatch, produce corrected payload, and select TRANSFORM_AND_RETRY.",
                "grader": True,
                "score_range": [0.01, 0.99]
            },
            {
                "task_id": "task_l3",
                "name": "Cascade root cause diagnosis", 
                "difficulty": "hard",
                "description": "Three linked tools - one timeout causes downstream failure. Agent must identify root cause.",
                "grader": True,
                "score_range": [0.01, 0.99]
            }
        ]
    }


@app.post("/grader")
async def grade_task(request: Dict[str, Any]) -> dict:
    """Grade a task based on action and scenario.
    
    Args:
        request: Dictionary containing task_id, action, and scenario
        
    Returns:
        Dictionary with grading result
    """
    try:
        task_id = request.get("task_id")
        action_data = request.get("action", {})
        scenario = request.get("scenario", {})
        
        # Convert action data to Action object
        action = Action(**action_data)
        
        if task_id == "task_l1":
            score = L1Grader.grade(action, scenario)
            return {
                "task_id": task_id,
                "score": score,
                "score_range": [0.01, 0.99]
            }
        elif task_id == "task_l2":
            score = L2Grader.grade(action, scenario)
            return {
                "task_id": task_id,
                "score": score,
                "score_range": [0.01, 0.99]
            }
        elif task_id == "task_l3":
            root_score, idem_score = L3Grader.grade(action, scenario)
            # For grader endpoint, return combined score
            combined_score = (root_score + idem_score) / 2
            return {
                "task_id": task_id,
                "score": combined_score,
                "score_range": [0.01, 0.99]
            }
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown task_id: {task_id}"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/reset")
async def reset(seed: Optional[int] = None) -> dict:
    """Reset the environment to initial state.
    
    Args:
        seed: Optional random seed for reproducibility
        
    Returns:
        Dictionary with observation
    """
    try:
        if seed is None:
            seed = 42
        
        observation = episode_manager.reset(seed)
        
        return {
            "observation": observation.model_dump(),
            "info": {"episode_id": episode_manager.episode_id},
        }
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "error": str(e),
                "observation": None,
                "info": {},
            },
        )


@app.post("/step")
async def step(action: Action) -> dict:
    """Execute one step in the environment.
    
    Args:
        action: The agent's action
        
    Returns:
        Dictionary with observation, reward, done, and info
    """
    try:
        observation, reward, done, info = episode_manager.step(action)
        
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "error": str(e),
                "observation": None,
                "reward": None,
                "done": False,
                "info": {},
            },
        )


@app.get("/state")
async def get_state() -> dict:
    """Get current episode state.
    
    Returns:
        Dictionary with episode state
    """
    try:
        state = episode_manager.state()
        return state.model_dump()
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "error": str(e),
                "state": None,
            },
        )


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> dict:
    """OpenEnv metadata endpoint."""
    return {
        "name": "agentic-dlq-triage",
        "description": "OpenEnv environment for training RL agents to diagnose and recover from production tool-call failures.",
        "version": "1.0.0",
        "author": "Pulagalla Sai Kiran",
        "tags": ["agentic", "reliability", "tool-calling", "error-recovery"],
    }


@app.get("/schema")
async def schema() -> dict:
    """OpenEnv schema endpoint — returns action, observation, and state schemas."""
    return {
        "observation": Observation.model_json_schema(),
        "action": Action.model_json_schema(),
        "state": EpisodeState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: dict) -> dict:
    """MCP JSON-RPC endpoint."""
    return {
        "jsonrpc": "2.0",
        "id": request.get("id", 1),
        "result": {
            "name": "agentic-dlq-triage",
            "version": "1.0.0",
            "capabilities": ["reset", "step", "state"],
        },
    }


def main():
    """Main entry point for server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
