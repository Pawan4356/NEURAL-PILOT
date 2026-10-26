def main() -> None:
    import logging
    import os

    import uvicorn

    from . import config, observability

    logging.basicConfig(
        level=os.environ.get("NEURALPILOT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.info(
        "Loaded %d setting(s) from %s" % (len(config.DOTENV_LOADED), config.DOTENV_PATH)
        if config.DOTENV_LOADED
        else "No .env settings loaded (%s)" % config.DOTENV_PATH
    )
    logging.info(
        "LLM source: %s, backend: %s (model %s)",
        config.LLM_SOURCE,
        config.LLM_BACKEND,
        config.active_llm_model(),
    )
    logging.info(observability.tracing_status())
    logging.info(
        "MLflow tracking: %s",
        f"on ({config.MLFLOW_TRACKING_URI})" if config.MLFLOW_ENABLED else "off",
    )
    logging.info("Candidates per iteration: %d", config.MAX_CANDIDATES)
    uvicorn.run(
        "neuralpilot.web.app:app",
        host=os.environ.get("NEURALPILOT_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEURALPILOT_PORT", "8000")),
    )
