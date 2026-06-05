"""Configuration loading contracts for the ROGII baseline project."""

from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when project configuration is invalid."""


def require_existing_file(path: str | Path) -> Path:
    """Return a path if it exists, otherwise raise a configuration error."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigError(f"Config file does not exist: {resolved}")
    return resolved


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to load config files. Install with: pip install pyyaml") from exc

    resolved = require_existing_file(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {resolved}")
    return data
