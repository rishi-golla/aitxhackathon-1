"""
Configuration Management System.

Centralized configuration with validation using Pydantic.
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml
import structlog
from pydantic import BaseModel, Field, field_validator

log = structlog.get_logger()


class CameraConfig(BaseModel):
    """Configuration for a camera source."""
    source: str | int
    name: str
    fps: int = Field(default=30, ge=1, le=120)
    resolution: tuple[int, int] = (1920, 1080)
    enabled: bool = True
    buffer_size: int = Field(default=10, ge=1, le=100)


class DetectionConfig(BaseModel):
    """Configuration for YOLO detection."""
    model: str = "yolov8s-worldv2"
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    device: str = "auto"
    classes: list[str] = []
    tensorrt: bool = False
    enable_tracking: bool = False

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices = {"auto", "cuda", "cpu", "mps"}
        if v not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")
        return v


class VLMConfig(BaseModel):
    """Configuration for Vision Language Model."""
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    quantization: str = Field(default="4bit", pattern=r"^(none|4bit|8bit)$")
    max_tokens: int = Field(default=512, ge=64, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    enabled: bool = True
    device_map: str = "auto"


class RAGConfig(BaseModel):
    """Configuration for RAG system."""
    documents_path: str = "data/osha_standards"
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = Field(default=3, ge=1, le=10)
    index_path: str = "data/osha_index"


class AlertConfig(BaseModel):
    """Configuration for alert system."""
    engine: str = "gtts"  # "piper", "edge", "gtts"
    voice: str = "en"
    rate: float = Field(default=1.0, ge=0.5, le=2.0)
    cooldown_seconds: int = Field(default=30, ge=5, le=300)
    enabled: bool = True
    cache_dir: str = ".alert_cache"


class DashboardConfig(BaseModel):
    """Configuration for Streamlit dashboard."""
    host: str = "0.0.0.0"
    port: int = Field(default=8501, ge=1024, le=65535)
    refresh_rate: float = Field(default=0.5, ge=0.1, le=5.0)
    theme: str = "dark"


class AgentConfig(BaseModel):
    """Configuration for agent framework."""
    perception_enabled: bool = True
    policy_enabled: bool = True
    coach_enabled: bool = True
    max_analysis_time_ms: int = Field(default=3000, ge=500, le=10000)
    model_backend: str = "local"  # "local", "bedrock", "openai"


class AppConfig(BaseModel):
    """Main application configuration."""
    cameras: list[CameraConfig] = []
    detection: DetectionConfig = DetectionConfig()
    vlm: VLMConfig = VLMConfig()
    rag: RAGConfig = RAGConfig()
    alerts: AlertConfig = AlertConfig()
    dashboard: DashboardConfig = DashboardConfig()
    agents: AgentConfig = AgentConfig()
    zones_config: str = "config/zones.yaml"
    database_path: str = "data/violations.db"
    log_level: str = "INFO"
    debug: bool = False

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper


class ConfigManager:
    """
    Singleton configuration manager.

    Features:
    - YAML file loading/saving
    - Environment variable overrides
    - Config validation
    - Hot reload support
    - Nested key access
    """

    _instance: Optional["ConfigManager"] = None

    def __init__(self, config_path: str = "config/app.yaml"):
        """
        Initialize config manager.

        Args:
            config_path: Path to configuration YAML file
        """
        self._config_path = Path(config_path)
        self._config: Optional[AppConfig] = None
        self._load()

    @classmethod
    def get_instance(cls, config_path: str = "config/app.yaml") -> "ConfigManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _load(self) -> None:
        """Load configuration from file."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r") as f:
                    data = yaml.safe_load(f) or {}

                # Apply environment variable overrides
                data = self._apply_env_overrides(data)

                # Parse and validate
                self._config = AppConfig(**data)
                log.info("config_loaded", path=str(self._config_path))

            except Exception as e:
                log.error("config_load_error", error=str(e))
                self._config = AppConfig()  # Use defaults
        else:
            log.warning("config_not_found_using_defaults", path=str(self._config_path))
            self._config = AppConfig()

    def _apply_env_overrides(self, data: dict) -> dict:
        """Apply environment variable overrides."""
        # Format: APP_SECTION_KEY=value
        # e.g., APP_DETECTION_CONFIDENCE=0.5

        prefix = "APP_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Parse key: APP_DETECTION_CONFIDENCE -> ["detection", "confidence"]
            parts = key[len(prefix):].lower().split("_")

            if len(parts) < 2:
                continue

            section = parts[0]
            setting = "_".join(parts[1:])

            # Get or create section
            if section not in data:
                data[section] = {}

            # Convert value type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

            data[section][setting] = value
            log.debug("env_override_applied", key=key, section=section, setting=setting)

        return data

    def load(self) -> AppConfig:
        """Reload and return configuration."""
        self._load()
        return self._config

    def save(self) -> None:
        """Save current configuration to file."""
        if self._config is None:
            return

        # Convert to dict
        data = self._config.model_dump()

        # Ensure directory exists
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        log.info("config_saved", path=str(self._config_path))

    def reload(self) -> AppConfig:
        """Reload configuration (hot reload)."""
        return self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "detection.confidence")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if self._config is None:
            return default

        parts = key.split(".")
        obj = self._config

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default

        return obj

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "detection.confidence")
            value: Value to set
        """
        if self._config is None:
            return

        parts = key.split(".")
        obj = self._config

        # Navigate to parent
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return

        # Set value
        final_key = parts[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
            log.debug("config_value_set", key=key, value=value)

    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            self._load()
        return self._config


def get_config() -> AppConfig:
    """Convenience function to get current config."""
    return ConfigManager.get_instance().config


def create_default_config(path: str = "config/app.yaml") -> None:
    """Create a default configuration file."""
    config = AppConfig(
        cameras=[
            CameraConfig(source=0, name="Main Camera"),
        ],
        detection=DetectionConfig(
            model="yolov8s-worldv2",
            confidence=0.3,
            classes=[
                "person", "hardhat", "safety vest", "safety glasses",
                "gloves", "bare hands", "forklift", "machinery"
            ]
        ),
        alerts=AlertConfig(
            engine="gtts",
            cooldown_seconds=30
        )
    )

    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    log.info("default_config_created", path=path)
