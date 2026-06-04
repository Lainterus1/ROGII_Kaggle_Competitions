"""Configuration loading contracts for the ROGII baseline project."""

from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when project configuration is invalid."""


def require_existing_file(path: str | Path) -> Path:
    """Return a path if it exists, otherwise raise a configuration error."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigError(f"Config file does not exist: {resolved}")
    return resolved
