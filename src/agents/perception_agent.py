"""
Perception Agent for scene analysis.

Analyzes factory scenes using VLM to understand context.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import structlog

from src.pipeline.vlm_analysis import VLMAnalyzer, SceneDescription, MockVLMAnalyzer
from src.pipeline.detection import Detection
from src.zones.zone_manager import SafetyZone

log = structlog.get_logger()


@dataclass
class PerceptionResult:
    """Result from perception agent analysis."""
    scene: SceneDescription
    analysis_time_ms: float
    detections_count: int
    zone_name: Optional[str]
    raw_detections: list[Detection]


class PerceptionAgent:
    """
    Perception agent that analyzes factory scenes.

    Uses VLM to understand scene context beyond raw detections.
    Identifies workers, their activities, and potential hazards.
    """

    def __init__(
        self,
        vlm_analyzer: Optional[VLMAnalyzer] = None,
        use_mock: bool = False,
        max_analysis_time_ms: int = 3000
    ):
        """
        Initialize perception agent.

        Args:
            vlm_analyzer: VLM analyzer instance
            use_mock: Use mock analyzer for testing
            max_analysis_time_ms: Maximum time for analysis
        """
        if use_mock:
            self._vlm = MockVLMAnalyzer()
        else:
            self._vlm = vlm_analyzer or VLMAnalyzer()

        self.max_analysis_time_ms = max_analysis_time_ms
        self._system_prompt = """You are a factory safety perception system.
Your job is to analyze camera frames and identify:
1. Workers visible in the scene
2. PPE each worker is wearing
3. Activities being performed
4. Potential hazards
5. Zone compliance

Be thorough but concise. Focus on safety-relevant observations."""

        log.info("perception_agent_initialized")

    def _format_zone_context(self, zone: Optional[SafetyZone]) -> str:
        """Format zone information for prompt."""
        if not zone:
            return "General work area - standard safety protocols apply."

        context_parts = [
            f"Zone: {zone.name}",
            f"Type: {zone.zone_type.value}"
        ]

        if zone.required_ppe:
            context_parts.append(f"Required PPE: {', '.join(zone.required_ppe)}")

        if zone.osha_reference:
            context_parts.append(f"OSHA Reference: {zone.osha_reference}")

        if zone.max_occupancy:
            context_parts.append(f"Max Occupancy: {zone.max_occupancy}")

        if zone.description:
            context_parts.append(f"Description: {zone.description}")

        return "\n".join(context_parts)

    async def analyze(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        zone: Optional[SafetyZone] = None
    ) -> PerceptionResult:
        """
        Analyze a frame for safety context.

        Args:
            frame: BGR image
            detections: YOLO detection results
            zone: Optional zone context

        Returns:
            PerceptionResult with scene understanding
        """
        start_time = time.time()

        # Format zone context
        zone_context = self._format_zone_context(zone)

        # Run VLM analysis
        try:
            # Add timeout
            scene = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._vlm.analyze_frame(frame, detections, zone_context)
                ),
                timeout=self.max_analysis_time_ms / 1000.0
            )
        except asyncio.TimeoutError:
            log.warning("perception_analysis_timeout", max_ms=self.max_analysis_time_ms)
            # Return basic scene from detections only
            scene = self._create_basic_scene(detections, zone)
        except Exception as e:
            log.error("perception_analysis_error", error=str(e))
            scene = self._create_basic_scene(detections, zone)

        analysis_time = (time.time() - start_time) * 1000

        result = PerceptionResult(
            scene=scene,
            analysis_time_ms=analysis_time,
            detections_count=len(detections),
            zone_name=zone.name if zone else None,
            raw_detections=detections
        )

        log.info(
            "perception_analysis_complete",
            workers=scene.workers_detected,
            concerns=len(scene.safety_concerns),
            time_ms=round(analysis_time, 1)
        )

        return result

    def _create_basic_scene(
        self,
        detections: list[Detection],
        zone: Optional[SafetyZone]
    ) -> SceneDescription:
        """Create basic scene description from detections only."""
        workers = [d for d in detections if d.class_name.lower() in {"person", "worker", "human"}]

        ppe_status = {}
        ppe_classes = {"hardhat", "safety vest", "safety glasses", "gloves", "face shield", "welding mask"}

        for i, worker in enumerate(workers):
            worker_id = f"worker_{i+1}"
            ppe_status[worker_id] = []

            for det in detections:
                if det.class_name.lower() in ppe_classes:
                    # Simple proximity check
                    wx1, wy1, wx2, wy2 = worker.bbox
                    dx1, dy1, dx2, dy2 = det.bbox

                    # Check overlap
                    if dx1 < wx2 and dx2 > wx1 and dy1 < wy2 and dy2 > wy1:
                        ppe_status[worker_id].append(det.class_name)

        concerns = []
        if zone and zone.required_ppe:
            for worker_id, ppe_list in ppe_status.items():
                ppe_normalized = [p.lower().replace("_", " ") for p in ppe_list]
                for required in zone.required_ppe:
                    if required.lower() not in ppe_normalized:
                        concerns.append(f"{worker_id} may be missing {required}")

        return SceneDescription(
            workers_detected=len(workers),
            activities=["working"] if workers else [],
            hazards_observed=[],
            ppe_status=ppe_status,
            zone_assessment=f"Automated analysis in {zone.name}" if zone else "Automated analysis",
            safety_concerns=concerns,
            confidence=0.6,
            inference_time_ms=0.0
        )

    def get_info(self) -> dict:
        """Get agent information."""
        return {
            "name": "PerceptionAgent",
            "max_analysis_time_ms": self.max_analysis_time_ms,
            "vlm_info": self._vlm.get_model_info() if hasattr(self._vlm, 'get_model_info') else {}
        }
