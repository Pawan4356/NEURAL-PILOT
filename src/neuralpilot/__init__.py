def main() -> None:
    import logging
    import os

    import uvicorn

    from . import config, observability

    logging.basicConfig(
        level=os.environ.get("NEURALPILOT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.info("LLM backend: %s (model %s)", config.LLM_BACKEND, config.LLM_CONFIG.model)
    logging.info(observability.tracing_status())
    uvicorn.run(
        "neuralpilot.web.app:app",
        host=os.environ.get("NEURALPILOT_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEURALPILOT_PORT", "8000")),
    )
