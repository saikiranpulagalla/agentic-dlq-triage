"""Server entry point for OpenEnv deployment."""

from dlq_triage.main import app


def main():
    """Main entry point for server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()