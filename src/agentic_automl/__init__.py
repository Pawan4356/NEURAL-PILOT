def main() -> None:
    import logging
    import os

    import uvicorn

    logging.basicConfig(
        level=os.environ.get("AUTOML_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "agentic_automl.web.app:app",
        host=os.environ.get("AUTOML_HOST", "127.0.0.1"),
        port=int(os.environ.get("AUTOML_PORT", "8000")),
    )
