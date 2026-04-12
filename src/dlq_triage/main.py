"""FastAPI application for AgenticDLQ Triage environment."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional, Any
from pydantic import BaseModel
from dlq_triage.models import Observation, Action, Reward, EpisodeState
from dlq_triage.episode import EpisodeManager

class ResetRequest(BaseModel):
    seed: Optional[int] = 42

app = FastAPI(
    title="AgenticDLQ Triage",
    description="OpenEnv environment for diagnosing tool-call failures",
    version="1.0.0",
)

# Global episode manager
episode_manager = EpisodeManager()


@app.post("/reset")
async def reset(request: Request) -> dict:
    """Reset the environment to initial state.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Dictionary with observation
    """
    try:
        seed = 42  # Default seed
        
        # Try to parse JSON body
        try:
            body = await request.json()
            if isinstance(body, dict) and "seed" in body:
                seed = int(body["seed"])
        except:
            # If no JSON body or parsing fails, use default seed
            pass
        
        observation = episode_manager.reset(seed)
        
        return {
            "observation": observation.model_dump(),
            "info": {"episode_id": episode_manager.episode_id, "seed": seed},
        }
    except Exception as e:
        print(f"Reset endpoint error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
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
            status_code=500,
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
            status_code=500,
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
