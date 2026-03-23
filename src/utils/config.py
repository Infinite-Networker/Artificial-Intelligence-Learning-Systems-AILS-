"""
AILS Configuration Manager
Centralized configuration for all AILS modules.
Created by Cherry Computer Ltd.
"""

import os
import yaml
import logging
from typing import Any, Dict, Optional


class AILSConfig:
    """
    AILS Centralized Configuration Manager.
    Loads from YAML config file and environment variables.

    Example:
        config = AILSConfig()
        db_host = config.get("database.host", "localhost")
    """

    DEFAULTS = {
        "database": {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "",
            "name": "AILS_data",
        },
        "scraper": {
            "timeout": 30,
            "rate_limit": 1.0,
            "max_retries": 3,
        },
        "model": {
            "hidden_units": [128, 64],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
        },
        "ethics": {
            "disparity_threshold": 0.1,
            "impact_threshold": 0.8,
            "privacy_epsilon": 1.0,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        self._config = self._deep_copy(self.DEFAULTS)
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        self._load_from_env()
        self._setup_logging()

    def _deep_copy(self, d: Dict) -> Dict:
        import copy
        return copy.deepcopy(d)

    def _load_from_file(self, path: str) -> None:
        with open(path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        self._deep_merge(self._config, file_config)

    def _load_from_env(self) -> None:
        """Load overrides from environment variables."""
        env_map = {
            "AILS_DB_HOST":     ("database", "host"),
            "AILS_DB_PORT":     ("database", "port"),
            "AILS_DB_USER":     ("database", "user"),
            "AILS_DB_PASSWORD": ("database", "password"),
            "AILS_DB_NAME":     ("database", "name"),
            "AILS_API_PORT":    ("api", "port"),
            "AILS_LOG_LEVEL":   ("logging", "level"),
        }
        for env_key, (section, key) in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                self._config[section][key] = val

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def _setup_logging(self) -> None:
        log_cfg = self._config.get("logging", {})
        logging.basicConfig(
            level=getattr(logging, log_cfg.get("level", "INFO")),
            format=log_cfg.get(
                "format",
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
        )

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a config value using dot notation.

        Example:
            config.get("database.host")
            config.get("model.learning_rate", 0.001)
        """
        parts = key_path.split(".")
        current = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key_path: str, value: Any) -> None:
        """Set a config value using dot notation."""
        parts = key_path.split(".")
        current = self._config
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    def to_dict(self) -> Dict:
        return self._deep_copy(self._config)


# Global config instance
config = AILSConfig()
