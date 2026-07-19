"""Shared rich + file logging setup used by train.py and evaluate_translations.py."""
import logging
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()

LOGGER_NAME = "translategemma"
logger = logging.getLogger(LOGGER_NAME)


def _level_from_str(name):
    return getattr(logging, str(name).upper(), logging.INFO)


def _resolve_run_name(config, run_name=None):
    if run_name:
        return run_name
    out_dir = config.get("model", {}).get("output_dir", "./run")
    base = str(out_dir).rstrip("/").split("/")[-1]
    return base or "run"


def setup_logging(config, run_name=None, logs_dir=None, console_level=None, file_level=None):
    """Configure rich console logging + full-level file logging.

    File name is run-based: <logs_dir>/<timestamp>_<run_name>.log
    Returns the path to the log file.
    """
    log_cfg = config.get("logging", {}) or {}
    logs_dir = logs_dir or log_cfg.get("logs_dir", "logs")
    console_level = _level_from_str(console_level or log_cfg.get("level", "INFO"))
    file_level = _level_from_str(file_level or log_cfg.get("file_level", "DEBUG"))

    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    run = _resolve_run_name(config, run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{run}.log"
    log_path = Path(logs_dir) / log_filename

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        level=console_level,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root.addHandler(rich_handler)
    root.addHandler(file_handler)

    # Tame noisy third-party loggers; still captured by file handler at file_level.
    for name, lvl in [
        ("transformers", logging.WARNING),
        ("datasets", logging.WARNING),
        ("peft", logging.INFO),
        ("trl", logging.INFO),
        ("torch", logging.WARNING),
        ("bitsandbytes", logging.WARNING),
        ("unbabelcomet", logging.WARNING),
        ("comet", logging.WARNING),
    ]:
        logging.getLogger(name).setLevel(lvl)

    logger.info(f"Logging initialized. Console level={logging.getLevelName(console_level)}, "
                f"file level={logging.getLevelName(file_level)}")
    logger.info(f"Log file: [bold]{log_path}[/bold]")
    return log_path


def log_config_summary(config):
    table = Table(title="Configuration Summary", show_lines=False, header_style="bold cyan")
    table.add_column("Section", style="bold magenta")
    table.add_column("Key", style="white")
    table.add_column("Value", style="green")
    for section, values in config.items():
        if not isinstance(values, dict):
            table.add_row(str(section), "-", str(values))
            continue
        for k, v in values.items():
            table.add_row(str(section), str(k), str(v))
    console.print(Panel(table, border_style="cyan"))


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
