"""
Safety Zone Manager for factory environments.

Defines and checks safety zones with PPE requirements.
"""

import json
import yaml
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import structlog

log = structlog.get_logger()

# Try importing shapely for geometry operations
try:
    from shapely.geometry import Polygon, Point, box
    from shapely.strtree import STRtree
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    log.warning("shapely_not_installed")


class ZoneType(Enum):
    """Types of safety zones."""
    PPE_REQUIRED = "ppe_required"  # Specific PPE needed
    RESTRICTED = "restricted"  # Authorized personnel only
    HAZARD = "hazard"  # Active hazard area
    SAFE = "safe"  # General work area
    MACHINE = "machine"  # Machine operation zone


@dataclass
class SafetyZone:
    """Definition of a safety zone."""
    zone_id: str
    name: str
    zone_type: ZoneType
    polygon: list[tuple[int, int]]  # Vertex coordinates [(x,y), ...]
    required_ppe: list[str] = field(default_factory=list)  # ["hardhat", "safety_vest"]
    max_occupancy: Optional[int] = None
    active_hours: Optional[tuple[int, int]] = None  # (start_hour, end_hour) 24h format
    osha_reference: Optional[str] = None  # "29 CFR 1910.212"
    description: Optional[str] = None
    enabled: bool = True


@dataclass
class ZoneViolation:
    """A detected zone violation."""
    zone: SafetyZone
    violation_type: str  # "missing_ppe", "unauthorized", "overcrowded"
    worker_id: Optional[str]
    missing_items: list[str]
    timestamp: float
    frame_id: int
    confidence: float
    bbox: Optional[tuple[int, int, int, int]] = None  # Worker bounding box


class ZoneManager:
    """
    Manages safety zones and checks for violations.

    Features:
    - Polygon-based zone definitions
    - PPE requirement checking
    - Spatial indexing for fast lookups
    - Time-based zone rules
    - YAML configuration
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize zone manager.

        Args:
            config_path: Path to zone configuration YAML file
        """
        if not SHAPELY_AVAILABLE:
            log.warning("shapely_not_available_using_fallback")

        self._zones: dict[str, SafetyZone] = {}
        self._polygons: dict[str, Polygon] = {} if SHAPELY_AVAILABLE else {}
        self._spatial_index: Optional[STRtree] = None
        self._index_dirty = True

        if config_path:
            self.load_config(config_path)

    def _build_spatial_index(self) -> None:
        """Rebuild spatial index for fast lookups."""
        if not SHAPELY_AVAILABLE or not self._polygons:
            return

        geometries = list(self._polygons.values())
        if geometries:
            self._spatial_index = STRtree(geometries)
        self._index_dirty = False

    def add_zone(self, zone: SafetyZone) -> None:
        """Add a safety zone."""
        self._zones[zone.zone_id] = zone

        if SHAPELY_AVAILABLE:
            self._polygons[zone.zone_id] = Polygon(zone.polygon)

        self._index_dirty = True
        log.info("zone_added", zone_id=zone.zone_id, name=zone.name)

    def remove_zone(self, zone_id: str) -> None:
        """Remove a safety zone."""
        if zone_id in self._zones:
            del self._zones[zone_id]
            if SHAPELY_AVAILABLE and zone_id in self._polygons:
                del self._polygons[zone_id]
            self._index_dirty = True
            log.info("zone_removed", zone_id=zone_id)

    def get_zone(self, zone_id: str) -> Optional[SafetyZone]:
        """Get a zone by ID."""
        return self._zones.get(zone_id)

    def list_zones(self) -> list[SafetyZone]:
        """List all zones."""
        return list(self._zones.values())

    def check_point(self, x: int, y: int) -> Optional[SafetyZone]:
        """
        Find which zone contains a point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            SafetyZone if point is in a zone, None otherwise
        """
        if not SHAPELY_AVAILABLE:
            return self._check_point_fallback(x, y)

        if self._index_dirty:
            self._build_spatial_index()

        point = Point(x, y)

        # Check all zones
        for zone_id, polygon in self._polygons.items():
            zone = self._zones[zone_id]
            if zone.enabled and polygon.contains(point):
                return zone

        return None

    def _check_point_fallback(self, x: int, y: int) -> Optional[SafetyZone]:
        """Fallback point-in-polygon without shapely."""
        for zone in self._zones.values():
            if not zone.enabled:
                continue
            if self._point_in_polygon(x, y, zone.polygon):
                return zone
        return None

    @staticmethod
    def _point_in_polygon(x: int, y: int, polygon: list[tuple[int, int]]) -> bool:
        """Ray casting algorithm for point-in-polygon."""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def get_zones_for_bbox(
        self,
        bbox: tuple[int, int, int, int]
    ) -> list[SafetyZone]:
        """
        Get all zones that intersect with a bounding box.

        Args:
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            List of intersecting zones
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Check center point
        zone = self.check_point(center_x, center_y)
        if zone:
            return [zone]

        # Check bottom center (feet position)
        zone = self.check_point(center_x, y2)
        if zone:
            return [zone]

        return []

    def _associate_ppe_with_workers(
        self,
        detections: list
    ) -> dict[str, list[str]]:
        """
        Associate PPE items with workers based on spatial proximity.

        Args:
            detections: List of Detection objects

        Returns:
            Dict mapping worker_id to list of PPE items they have
        """
        from src.pipeline.detection import Detection

        # Separate workers and PPE
        workers = []
        ppe_items = []

        worker_classes = {"person", "worker", "human"}
        ppe_classes = {
            "hardhat", "safety helmet", "hard hat", "helmet",
            "safety glasses", "goggles", "face shield", "welding mask",
            "safety vest", "high visibility vest", "reflective vest",
            "gloves", "safety gloves", "work gloves"
        }

        for det in detections:
            class_lower = det.class_name.lower()
            if class_lower in worker_classes:
                workers.append(det)
            elif class_lower in ppe_classes:
                ppe_items.append(det)

        # Associate PPE with closest worker
        worker_ppe: dict[str, list[str]] = {}

        for worker in workers:
            worker_id = str(worker.track_id) if worker.track_id else f"w_{hash(worker.bbox)}"
            worker_ppe[worker_id] = []

            wx1, wy1, wx2, wy2 = worker.bbox
            worker_center = ((wx1 + wx2) / 2, (wy1 + wy2) / 2)
            worker_width = wx2 - wx1
            worker_height = wy2 - wy1

            for ppe in ppe_items:
                px1, py1, px2, py2 = ppe.bbox
                ppe_center = ((px1 + px2) / 2, (py1 + py2) / 2)

                # Check if PPE overlaps or is near worker
                # PPE should be within worker bbox or slightly outside
                overlap_x = px1 < wx2 and px2 > wx1
                overlap_y = py1 < wy2 and py2 > wy1

                # Or within reasonable distance (1.5x worker size)
                dist_x = abs(ppe_center[0] - worker_center[0])
                dist_y = abs(ppe_center[1] - worker_center[1])

                if (overlap_x and overlap_y) or (
                    dist_x < worker_width * 0.75 and dist_y < worker_height * 0.75
                ):
                    # Normalize PPE name
                    ppe_name = self._normalize_ppe_name(ppe.class_name)
                    if ppe_name not in worker_ppe[worker_id]:
                        worker_ppe[worker_id].append(ppe_name)

        return worker_ppe

    @staticmethod
    def _normalize_ppe_name(class_name: str) -> str:
        """Normalize PPE class names to standard names."""
        name_lower = class_name.lower()

        # Head protection
        if any(x in name_lower for x in ["hardhat", "hard hat", "helmet", "safety helmet"]):
            return "hardhat"

        # Eye protection
        if any(x in name_lower for x in ["glasses", "goggles"]):
            return "safety_glasses"
        if "face shield" in name_lower:
            return "face_shield"
        if "welding" in name_lower:
            return "welding_mask"

        # Body protection
        if any(x in name_lower for x in ["vest", "high vis", "reflective"]):
            return "safety_vest"

        # Hand protection
        if "glove" in name_lower:
            return "gloves"

        return class_name.lower().replace(" ", "_")

    def check_detection(
        self,
        detection,  # Detection object
        all_detections: list,
        frame_id: int = 0,
        timestamp: float = 0.0
    ) -> Optional[ZoneViolation]:
        """
        Check if a detection violates any zone rules.

        Args:
            detection: The detection to check (should be a person/worker)
            all_detections: All detections in the frame
            frame_id: Current frame ID
            timestamp: Current timestamp

        Returns:
            ZoneViolation if violation found, None otherwise
        """
        from src.pipeline.detection import Detection

        # Only check person detections
        if detection.class_name.lower() not in {"person", "worker", "human"}:
            return None

        # Find which zone the worker is in
        x1, y1, x2, y2 = detection.bbox
        center_x = (x1 + x2) // 2
        foot_y = y2  # Use feet position

        zone = self.check_point(center_x, foot_y)
        if not zone or not zone.enabled:
            return None

        # Check time-based rules
        if zone.active_hours:
            import time
            current_hour = time.localtime().tm_hour
            start_hour, end_hour = zone.active_hours
            if not (start_hour <= current_hour < end_hour):
                return None  # Zone not active

        # Get worker's PPE
        worker_ppe = self._associate_ppe_with_workers(all_detections)
        worker_id = str(detection.track_id) if detection.track_id else f"w_{hash(detection.bbox)}"
        worker_items = worker_ppe.get(worker_id, [])

        # Check PPE requirements
        if zone.zone_type == ZoneType.PPE_REQUIRED and zone.required_ppe:
            missing = []
            for required in zone.required_ppe:
                req_normalized = self._normalize_ppe_name(required)
                if req_normalized not in worker_items:
                    missing.append(required)

            if missing:
                return ZoneViolation(
                    zone=zone,
                    violation_type="missing_ppe",
                    worker_id=worker_id,
                    missing_items=missing,
                    timestamp=timestamp or __import__("time").time(),
                    frame_id=frame_id,
                    confidence=detection.confidence,
                    bbox=detection.bbox
                )

        # Check restricted zones
        if zone.zone_type == ZoneType.RESTRICTED:
            return ZoneViolation(
                zone=zone,
                violation_type="restricted_area",
                worker_id=worker_id,
                missing_items=[],
                timestamp=timestamp or __import__("time").time(),
                frame_id=frame_id,
                confidence=detection.confidence,
                bbox=detection.bbox
            )

        return None

    def check_frame(
        self,
        detections: list,
        frame_id: int = 0,
        timestamp: float = 0.0
    ) -> list[ZoneViolation]:
        """
        Check all detections against all zones.

        Args:
            detections: All detections in frame
            frame_id: Current frame ID
            timestamp: Current timestamp

        Returns:
            List of violations found
        """
        violations = []

        for detection in detections:
            violation = self.check_detection(
                detection, detections, frame_id, timestamp
            )
            if violation:
                violations.append(violation)

        # Check occupancy
        zone_counts: dict[str, int] = {}
        for detection in detections:
            if detection.class_name.lower() in {"person", "worker", "human"}:
                x1, y1, x2, y2 = detection.bbox
                zone = self.check_point((x1 + x2) // 2, y2)
                if zone:
                    zone_counts[zone.zone_id] = zone_counts.get(zone.zone_id, 0) + 1

        for zone_id, count in zone_counts.items():
            zone = self._zones.get(zone_id)
            if zone and zone.max_occupancy and count > zone.max_occupancy:
                violations.append(ZoneViolation(
                    zone=zone,
                    violation_type="overcrowded",
                    worker_id=None,
                    missing_items=[],
                    timestamp=timestamp or __import__("time").time(),
                    frame_id=frame_id,
                    confidence=1.0
                ))

        return violations

    def save_config(self, path: str) -> None:
        """Save zone configuration to YAML file."""
        config = {"zones": []}

        for zone in self._zones.values():
            zone_dict = {
                "zone_id": zone.zone_id,
                "name": zone.name,
                "zone_type": zone.zone_type.value,
                "polygon": zone.polygon,
                "required_ppe": zone.required_ppe,
                "enabled": zone.enabled
            }

            if zone.max_occupancy:
                zone_dict["max_occupancy"] = zone.max_occupancy
            if zone.active_hours:
                zone_dict["active_hours"] = list(zone.active_hours)
            if zone.osha_reference:
                zone_dict["osha_reference"] = zone.osha_reference
            if zone.description:
                zone_dict["description"] = zone.description

            config["zones"].append(zone_dict)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        log.info("zone_config_saved", path=path, num_zones=len(self._zones))

    def load_config(self, path: str) -> None:
        """Load zone configuration from YAML file."""
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)

            if not config or "zones" not in config:
                log.warning("empty_zone_config", path=path)
                return

            self._zones.clear()
            self._polygons.clear()

            for zone_data in config["zones"]:
                # Convert polygon to list of tuples
                polygon = [tuple(p) for p in zone_data["polygon"]]

                zone = SafetyZone(
                    zone_id=zone_data["zone_id"],
                    name=zone_data["name"],
                    zone_type=ZoneType(zone_data["zone_type"]),
                    polygon=polygon,
                    required_ppe=zone_data.get("required_ppe", []),
                    max_occupancy=zone_data.get("max_occupancy"),
                    active_hours=tuple(zone_data["active_hours"]) if zone_data.get("active_hours") else None,
                    osha_reference=zone_data.get("osha_reference"),
                    description=zone_data.get("description"),
                    enabled=zone_data.get("enabled", True)
                )
                self.add_zone(zone)

            log.info("zone_config_loaded", path=path, num_zones=len(self._zones))

        except FileNotFoundError:
            log.warning("zone_config_not_found", path=path)
        except Exception as e:
            log.error("zone_config_load_error", path=path, error=str(e))
            raise


def draw_zones(
    frame,
    zones: list[SafetyZone],
    violations: Optional[list[ZoneViolation]] = None
):
    """
    Draw zone boundaries on frame.

    Args:
        frame: BGR image (numpy array)
        zones: List of zones to draw
        violations: Optional list of current violations to highlight
    """
    import cv2
    import numpy as np

    violation_zone_ids = set()
    if violations:
        violation_zone_ids = {v.zone.zone_id for v in violations}

    overlay = frame.copy()

    for zone in zones:
        if not zone.enabled:
            continue

        pts = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))

        # Color based on zone type and violation status
        if zone.zone_id in violation_zone_ids:
            color = (0, 0, 255)  # Red for active violation
            alpha = 0.3
        elif zone.zone_type == ZoneType.HAZARD:
            color = (0, 165, 255)  # Orange
            alpha = 0.2
        elif zone.zone_type == ZoneType.RESTRICTED:
            color = (0, 0, 255)  # Red
            alpha = 0.15
        elif zone.zone_type == ZoneType.PPE_REQUIRED:
            color = (0, 255, 255)  # Yellow
            alpha = 0.15
        else:
            color = (0, 255, 0)  # Green
            alpha = 0.1

        # Fill zone
        cv2.fillPoly(overlay, [pts], color)

        # Draw border
        cv2.polylines(frame, [pts], True, color, 2)

        # Label
        centroid = np.mean(zone.polygon, axis=0).astype(int)
        cv2.putText(
            frame,
            zone.name,
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Blend overlay
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    return frame
