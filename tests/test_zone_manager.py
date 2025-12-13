"""
Tests for Zone Manager.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.zones.zone_manager import (
    ZoneManager, SafetyZone, ZoneType, ZoneViolation
)


class TestZoneManager:
    """Test suite for ZoneManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ZoneManager()

        # Add test zones
        self.welding_zone = SafetyZone(
            zone_id="welding_bay",
            name="Welding Bay A",
            zone_type=ZoneType.PPE_REQUIRED,
            polygon=[(100, 100), (400, 100), (400, 400), (100, 400)],
            required_ppe=["hardhat", "safety_glasses", "welding_mask"],
            osha_reference="29 CFR 1910.252"
        )

        self.restricted_zone = SafetyZone(
            zone_id="electrical",
            name="Electrical Room",
            zone_type=ZoneType.RESTRICTED,
            polygon=[(500, 100), (700, 100), (700, 300), (500, 300)],
            required_ppe=["hardhat"],
            osha_reference="29 CFR 1910.303"
        )

        self.manager.add_zone(self.welding_zone)
        self.manager.add_zone(self.restricted_zone)

    def test_add_zone(self):
        """Test adding zones."""
        zones = self.manager.list_zones()
        assert len(zones) == 2
        assert self.manager.get_zone("welding_bay") is not None

    def test_remove_zone(self):
        """Test removing zones."""
        self.manager.remove_zone("welding_bay")
        assert self.manager.get_zone("welding_bay") is None
        assert len(self.manager.list_zones()) == 1

    def test_check_point_inside(self):
        """Test point inside zone."""
        # Point inside welding bay
        zone = self.manager.check_point(200, 200)
        assert zone is not None
        assert zone.zone_id == "welding_bay"

    def test_check_point_outside(self):
        """Test point outside all zones."""
        zone = self.manager.check_point(50, 50)
        assert zone is None

    def test_check_point_restricted(self):
        """Test point in restricted zone."""
        zone = self.manager.check_point(600, 200)
        assert zone is not None
        assert zone.zone_type == ZoneType.RESTRICTED

    def test_normalize_ppe_name(self):
        """Test PPE name normalization."""
        assert self.manager._normalize_ppe_name("Hard Hat") == "hardhat"
        assert self.manager._normalize_ppe_name("safety glasses") == "safety_glasses"
        assert self.manager._normalize_ppe_name("High Visibility Vest") == "safety_vest"


class TestSafetyZone:
    """Test SafetyZone dataclass."""

    def test_create_zone(self):
        """Test zone creation."""
        zone = SafetyZone(
            zone_id="test",
            name="Test Zone",
            zone_type=ZoneType.SAFE,
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)]
        )
        assert zone.zone_id == "test"
        assert zone.enabled is True

    def test_zone_with_ppe(self):
        """Test zone with PPE requirements."""
        zone = SafetyZone(
            zone_id="test",
            name="Test Zone",
            zone_type=ZoneType.PPE_REQUIRED,
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            required_ppe=["hardhat", "vest"]
        )
        assert len(zone.required_ppe) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
