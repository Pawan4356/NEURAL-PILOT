def main() -> None:
    import os

    import uvicorn

    uvicorn.run(
        "agentic_automl.web.app:app",
        host=os.environ.get("AUTOML_HOST", "127.0.0.1"),
        port=int(os.environ.get("AUTOML_PORT", "8000")),
    )
