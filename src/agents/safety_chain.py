"""
Safety Agent Chain - Orchestrates perception, policy, and coach agents.

Main pipeline: Frame -> Perception -> Policy -> Coach -> Alert
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import structlog

from src.pipeline.detection import Detection
from src.pipeline.vlm_analysis import SceneDescription
from src.zones.zone_manager import SafetyZone, ZoneViolation
from src.agents.perception_agent import PerceptionAgent, PerceptionResult
from src.agents.policy_agent import PolicyAgent, PolicyDecision
from src.agents.coach_agent import CoachAgent, CoachFeedback

log = structlog.get_logger()


@dataclass
class ChainResult:
    """Result from the full agent chain."""
    perception: PerceptionResult
    decision: PolicyDecision
    feedback: Optional[CoachFeedback]
    total_time_ms: float
    is_violation: bool
    violation: Optional[ZoneViolation] = None


@dataclass
class ChainStats:
    """Statistics for chain performance."""
    total_runs: int = 0
    violations_detected: int = 0
    avg_time_ms: float = 0.0
    avg_perception_ms: float = 0.0
    avg_policy_ms: float = 0.0
    avg_coach_ms: float = 0.0


class SafetyAgentChain:
    """
    Orchestrates the three-agent safety pipeline.

    Pipeline:
    1. Perception Agent: Analyzes scene with VLM
    2. Policy Agent: Evaluates against safety rules + OSHA
    3. Coach Agent: Generates constructive feedback

    Features:
    - Async execution
    - Timeout handling
    - Graceful degradation
    - Performance tracking
    """

    def __init__(
        self,
        perception: Optional[PerceptionAgent] = None,
        policy: Optional[PolicyAgent] = None,
        coach: Optional[CoachAgent] = None,
        max_chain_time_ms: int = 5000,
        skip_feedback_on_compliant: bool = True
    ):
        """
        Initialize safety chain.

        Args:
            perception: Perception agent instance
            policy: Policy agent instance
            coach: Coach agent instance
            max_chain_time_ms: Maximum time for full chain
            skip_feedback_on_compliant: Skip coach for compliant scenes
        """
        self.perception = perception or PerceptionAgent(use_mock=True)
        self.policy = policy or PolicyAgent()
        self.coach = coach or CoachAgent()
        self.max_chain_time_ms = max_chain_time_ms
        self.skip_feedback_on_compliant = skip_feedback_on_compliant

        self._stats = ChainStats()
        self._time_history: list[float] = []

        log.info("safety_chain_initialized", max_time_ms=max_chain_time_ms)

    async def process_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        zone: SafetyZone,
        frame_id: int = 0,
        timestamp: float = 0.0
    ) -> ChainResult:
        """
        Process a frame through the full agent pipeline.

        Args:
            frame: BGR image
            detections: YOLO detection results
            zone: Zone context
            frame_id: Frame identifier
            timestamp: Frame timestamp

        Returns:
            ChainResult with all agent outputs
        """
        start_time = time.time()
        timestamp = timestamp or time.time()

        # Stage 1: Perception
        try:
            perception_result = await asyncio.wait_for(
                self.perception.analyze(frame, detections, zone),
                timeout=(self.max_chain_time_ms / 1000.0) * 0.6
            )
        except asyncio.TimeoutError:
            log.warning("perception_timeout")
            perception_result = self._create_fallback_perception(detections, zone)
        except Exception as e:
            log.error("perception_error", error=str(e))
            perception_result = self._create_fallback_perception(detections, zone)

        # Stage 2: Policy evaluation
        try:
            policy_decision = await asyncio.wait_for(
                self.policy.evaluate(perception_result.scene, zone),
                timeout=(self.max_chain_time_ms / 1000.0) * 0.2
            )
        except asyncio.TimeoutError:
            log.warning("policy_timeout")
            policy_decision = PolicyDecision(
                is_violation=False,
                violation_type=None,
                severity="info",
                explanation="Policy evaluation timed out"
            )
        except Exception as e:
            log.error("policy_error", error=str(e))
            policy_decision = PolicyDecision(
                is_violation=False,
                violation_type=None,
                severity="info",
                explanation=f"Policy error: {str(e)}"
            )

        # Stage 3: Coach feedback (if violation or not skipping)
        feedback = None
        if policy_decision.is_violation or not self.skip_feedback_on_compliant:
            try:
                feedback = await asyncio.wait_for(
                    self.coach.generate_feedback(policy_decision, zone),
                    timeout=(self.max_chain_time_ms / 1000.0) * 0.2
                )
            except asyncio.TimeoutError:
                log.warning("coach_timeout")
            except Exception as e:
                log.error("coach_error", error=str(e))

        total_time = (time.time() - start_time) * 1000

        # Create violation object if needed
        violation = None
        if policy_decision.is_violation:
            violation = ZoneViolation(
                zone=zone,
                violation_type=policy_decision.violation_type or "unknown",
                worker_id=None,  # Could be extracted from perception
                missing_items=policy_decision.missing_items,
                timestamp=timestamp,
                frame_id=frame_id,
                confidence=policy_decision.confidence
            )

        # Update stats
        self._update_stats(
            total_time,
            perception_result.analysis_time_ms,
            policy_decision.evaluation_time_ms,
            policy_decision.is_violation
        )

        result = ChainResult(
            perception=perception_result,
            decision=policy_decision,
            feedback=feedback,
            total_time_ms=total_time,
            is_violation=policy_decision.is_violation,
            violation=violation
        )

        log.info(
            "chain_complete",
            is_violation=policy_decision.is_violation,
            severity=policy_decision.severity,
            total_ms=round(total_time, 1)
        )

        return result

    def _create_fallback_perception(
        self,
        detections: list[Detection],
        zone: SafetyZone
    ) -> PerceptionResult:
        """Create fallback perception result from detections only."""
        workers = [d for d in detections if d.class_name.lower() in {"person", "worker", "human"}]

        scene = SceneDescription(
            workers_detected=len(workers),
            activities=["working"] if workers else [],
            hazards_observed=[],
            ppe_status={},
            zone_assessment=f"Fallback analysis for {zone.name}",
            safety_concerns=[],
            confidence=0.5,
            inference_time_ms=0.0
        )

        return PerceptionResult(
            scene=scene,
            analysis_time_ms=0.0,
            detections_count=len(detections),
            zone_name=zone.name,
            raw_detections=detections
        )

    def _update_stats(
        self,
        total_ms: float,
        perception_ms: float,
        policy_ms: float,
        is_violation: bool
    ) -> None:
        """Update chain statistics."""
        self._stats.total_runs += 1
        if is_violation:
            self._stats.violations_detected += 1

        # Rolling average
        self._time_history.append(total_ms)
        if len(self._time_history) > 100:
            self._time_history = self._time_history[-100:]

        self._stats.avg_time_ms = sum(self._time_history) / len(self._time_history)

    async def process_batch(
        self,
        frames_data: list[tuple[np.ndarray, list[Detection], SafetyZone]]
    ) -> list[ChainResult]:
        """
        Process multiple frames concurrently.

        Args:
            frames_data: List of (frame, detections, zone) tuples

        Returns:
            List of ChainResults
        """
        tasks = []
        for i, (frame, detections, zone) in enumerate(frames_data):
            task = self.process_frame(frame, detections, zone, frame_id=i)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict:
        """Get chain statistics."""
        return {
            "total_runs": self._stats.total_runs,
            "violations_detected": self._stats.violations_detected,
            "violation_rate": (
                self._stats.violations_detected / self._stats.total_runs
                if self._stats.total_runs > 0 else 0
            ),
            "avg_time_ms": round(self._stats.avg_time_ms, 1),
            "max_chain_time_ms": self.max_chain_time_ms
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = ChainStats()
        self._time_history = []

    def get_agent_info(self) -> dict:
        """Get information about all agents."""
        return {
            "perception": self.perception.get_info(),
            "policy": self.policy.get_info(),
            "coach": self.coach.get_info(),
            "chain_stats": self.get_stats()
        }


class SimpleSafetyChain:
    """
    Simplified safety chain without VLM.

    Uses only YOLO detections and zone rules for faster processing.
    Suitable for real-time applications without GPU for VLM.
    """

    def __init__(self):
        self.policy = PolicyAgent()
        self.coach = CoachAgent()

    async def check_violations(
        self,
        detections: list[Detection],
        zone: SafetyZone,
        frame_id: int = 0,
        timestamp: float = 0.0
    ) -> tuple[Optional[PolicyDecision], Optional[CoachFeedback]]:
        """
        Quick violation check without VLM analysis.

        Args:
            detections: YOLO detections
            zone: Zone to check against
            frame_id: Frame ID
            timestamp: Timestamp

        Returns:
            Tuple of (PolicyDecision, CoachFeedback) if violation, else (None, None)
        """
        # Create basic scene from detections
        workers = [d for d in detections if d.class_name.lower() in {"person", "worker", "human"}]

        ppe_items = {}
        ppe_classes = {"hardhat", "safety vest", "safety glasses", "gloves", "face shield"}

        for i, worker in enumerate(workers):
            worker_id = f"worker_{i+1}"
            ppe_items[worker_id] = []

            for det in detections:
                if det.class_name.lower() in ppe_classes:
                    ppe_items[worker_id].append(det.class_name)

        scene = SceneDescription(
            workers_detected=len(workers),
            activities=[],
            hazards_observed=[],
            ppe_status=ppe_items,
            zone_assessment="Quick check",
            safety_concerns=[],
            confidence=0.7,
            inference_time_ms=0.0
        )

        # Evaluate
        decision = await self.policy.evaluate(scene, zone)

        if decision.is_violation:
            feedback = await self.coach.generate_feedback(decision, zone)
            return decision, feedback

        return None, None
