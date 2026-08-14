import os
import logging
import json
from datetime import datetime, timezone

# 1. Environment Variable Parsing & Configuration
try:
    PERCENTILE_CUTOFF = float(os.getenv("PERCENTILE_CUTOFF", "0.95"))
except (ValueError, TypeError):
    PERCENTILE_CUTOFF = 0.95

try:
    WIDTH_MULTIPLIER = float(os.getenv("WIDTH_MULTIPLIER", "2.0"))
except (ValueError, TypeError):
    WIDTH_MULTIPLIER = 2.0

try:
    DEFAULT_GSD = float(os.getenv("DEFAULT_GSD", "0.1"))
except (ValueError, TypeError):
    DEFAULT_GSD = 0.1


# 2. Structured JSON Logging Setup
class JsonFormatter(logging.Formatter):
    """
    Standard Library Formatter that serializes logs to JSON string format.
    """
    def format(self, record):
        log_payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        
        # Include exception tracebacks if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)
            
        # Extract dynamic keys passed in extra=dict(...)
        standard_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
            'processName', 'process'
        }
        for key, val in record.__dict__.items():
            if key not in standard_fields:
                log_payload[key] = val
                
        return json.dumps(log_payload)


def get_logger(name):
    """
    Retrieves or configures a logger instance with JsonFormatter attached to stream output.
    """
    logger = logging.getLogger(name)
    # Default log level can be set via env var, falling back to INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if initialized multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        
    return logger
