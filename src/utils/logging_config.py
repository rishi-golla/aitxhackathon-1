"""
Structured Logging Configuration.

Sets up structlog with appropriate processors for development and production.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = "logs/app.log",
    json_format: bool = False,
    include_timestamp: bool = True
) -> None:
    """
    Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        json_format: Output as JSON (for production)
        include_timestamp: Include timestamps in output
    """
    # Create log directory if needed
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

    # Define shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if include_timestamp:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    # Choose renderer based on format
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback
        )

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))

        # Use JSON format for file logs
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer()
            ]
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)

    # Console handler with pretty output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer
        ]
    )
    console_handler.setFormatter(console_formatter)

    # Replace default handlers
    root_logger = logging.getLogger()
    root_logger.handlers = [console_handler]
    if log_file:
        root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Optional logger name

    Returns:
        Configured logger
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


# Context manager for adding context to logs
class LogContext:
    """Context manager for adding temporary log context."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._token = None

    def __enter__(self):
        self._token = structlog.contextvars.bind_contextvars(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.kwargs.keys())


# Convenience functions
def log_info(event: str, **kwargs) -> None:
    """Log an info message."""
    structlog.get_logger().info(event, **kwargs)


def log_warning(event: str, **kwargs) -> None:
    """Log a warning message."""
    structlog.get_logger().warning(event, **kwargs)


def log_error(event: str, **kwargs) -> None:
    """Log an error message."""
    structlog.get_logger().error(event, **kwargs)


def log_debug(event: str, **kwargs) -> None:
    """Log a debug message."""
    structlog.get_logger().debug(event, **kwargs)
