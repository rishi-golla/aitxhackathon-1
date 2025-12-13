"""
Tests for Configuration Manager.
"""

import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import (
    ConfigManager, AppConfig, DetectionConfig, AlertConfig,
    create_default_config
)


class TestConfigManager:
    """Test suite for ConfigManager."""

    def setup_method(self):
        """Set up test fixtures."""
        ConfigManager.reset_instance()

    def test_default_config(self):
        """Test default configuration values."""
        config = AppConfig()

        assert config.detection.confidence == 0.3
        assert config.alerts.enabled is True
        assert config.log_level == "INFO"

    def test_detection_config_validation(self):
        """Test detection config validation."""
        # Valid confidence
        config = DetectionConfig(confidence=0.5)
        assert config.confidence == 0.5

        # Invalid confidence should raise
        with pytest.raises(ValueError):
            DetectionConfig(confidence=1.5)

    def test_alert_config_defaults(self):
        """Test alert config defaults."""
        config = AlertConfig()

        assert config.engine == "gtts"
        assert config.cooldown_seconds == 30
        assert config.enabled is True

    def test_create_default_config(self):
        """Test creating default config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test_config.yaml"
            create_default_config(path)

            assert Path(path).exists()

    def test_config_manager_singleton(self):
        """Test ConfigManager singleton pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/config.yaml"
            create_default_config(path)

            manager1 = ConfigManager.get_instance(path)
            manager2 = ConfigManager.get_instance(path)

            assert manager1 is manager2


class TestAppConfig:
    """Test AppConfig model."""

    def test_valid_log_level(self):
        """Test valid log levels."""
        config = AppConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"

        config = AppConfig(log_level="warning")
        assert config.log_level == "WARNING"

    def test_invalid_log_level(self):
        """Test invalid log level raises error."""
        with pytest.raises(ValueError):
            AppConfig(log_level="INVALID")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
