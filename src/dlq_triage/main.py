"""FastAPI application for AgenticDLQ Triage environment."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from dlq_triage.models import Observation, Action, Reward, EpisodeState
from dlq_triage.episode import EpisodeManager

app = FastAPI(
    title="AgenticDLQ Triage",
    description="OpenEnv environment for diagnosing tool-call failures",
    version="1.0.0",
)

# Global episode manager
episode_manager = EpisodeManager()


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
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
