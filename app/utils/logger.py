"""
Centralised logger — import `logger` from here everywhere in the project.

Uses loguru for structured, zero-config logging with file + console sinks.
"""

import sys
from loguru import logger
from app.core.config import LOG_LEVEL, LOG_FILE

# Remove the default loguru sink (plain stderr) so we can configure our own
logger.remove()

# ── Console sink ───────────────────────────────────────────────────────────────
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
)

# ── File sink (rotating, keeps last 3 runs) ────────────────────────────────────
logger.add(
    LOG_FILE,
    level="DEBUG",          # file always captures everything
    rotation="10 MB",
    retention=3,
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
)

__all__ = ["logger"]