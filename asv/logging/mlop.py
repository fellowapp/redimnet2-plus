"""MLOP logging adapter with safe fallback behavior."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _is_eer_metric(metric_key: str) -> bool:
    """True for primary ``.../eer/...`` keys and variant ``.../eer_<name>`` leaf segments."""
    normalized = metric_key.strip().strip("/")
    for part in normalized.split("/"):
        if part == "eer" or part.startswith("eer_"):
            return True
    return False


def _scaled_eer_value(value: Any) -> float | None:
    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        logger.warning("Unable to scale EER metric value to percentage: {}", value)
        return None


def init_mlop_logger(cfg: DictConfig, config_for_log: dict[str, Any]) -> Any | None:
    """Initialize mlop logger from env/config.

    Returns ``None`` when logging is disabled or setup is incomplete.
    """
    if str(cfg.logging.backend).lower() != "mlop":
        return None
    try:
        import mlop
    except Exception as exc:
        logger.warning("MLOP backend requested but import failed: {}", exc)
        return None

    project = str(cfg.logging.project)
    run_name = str(cfg.logging.run_name)
    api_key = _env("MLOP_API_KEY")
    org_id = _env("MLOPS_ORG_ID")
    auth_token = api_key or org_id
    if not auth_token:
        logger.warning("MLOP logging disabled: no MLOP_API_KEY or MLOPS_ORG_ID token provided.")
        return None
    if not api_key:
        logger.warning("MLOP_API_KEY is missing; using MLOPS_ORG_ID as auth token fallback.")
    elif not org_id:
        logger.debug("MLOPS_ORG_ID is missing; using MLOP_API_KEY as auth token.")

    settings = {
        "_auth": auth_token,
        "url_app": str(cfg.logging.url_app),
        "url_api": str(cfg.logging.url_api),
        "url_ingest": str(cfg.logging.url_ingest),
        "url_py": str(cfg.logging.url_py),
    }
    try:
        mlop.login(settings=settings, retry=False)
        run = mlop.init(
            project=project,
            name=run_name,
            config=config_for_log,
            settings=settings,
        )
        logger.info("MLOP logging started: project='{}' run='{}'", project, run_name)
        return run
    except Exception as exc:
        logger.warning("Failed to initialize MLOP logger, continuing without remote logging: {}", exc)
        return None


def log_mlop_metrics(run: Any | None, metrics: dict[str, Any]) -> None:
    if run is None:
        return
    sanitized_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if _is_eer_metric(key):
            scaled = _scaled_eer_value(value)
            if scaled is not None:
                sanitized_metrics[key] = scaled
            else:
                sanitized_metrics[key] = value
            continue
        sanitized_metrics[key] = value
    try:
        if sanitized_metrics:
            run.log(sanitized_metrics)
    except Exception as exc:
        logger.warning("MLOP metric logging failed: {}", exc)


def finish_mlop_run(run: Any | None) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        logger.warning("MLOP run finish failed: {}", exc)


def log_mlop_commit_artifact(run: Any | None, commit_hash: str, output_dir: str | Path) -> None:
    if run is None or not commit_hash:
        return
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        commit_file = out_dir / "git_commit.txt"
        commit_file.write_text(commit_hash + "\n", encoding="utf-8")
        run.log({"git/commit_hash": commit_hash})
    except Exception as exc:
        logger.warning("Failed to log git commit artifact to MLOP: {}", exc)
