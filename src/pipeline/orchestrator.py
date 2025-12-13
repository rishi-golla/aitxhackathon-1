"""
Real-Time Pipeline Orchestrator.

Integrates all components into a unified safety monitoring pipeline.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import numpy as np
import structlog

from src.pipeline.video_capture import VideoCapture, FramePacket
from src.pipeline.detection import PPEDetector, DetectionResult, Detection
from src.zones.zone_manager import ZoneManager, ZoneViolation
from src.agents.safety_chain import SafetyAgentChain, ChainResult
from src.tts.alert_system import AlertSystem, AlertManager
from src.utils.database import ViolationDatabase
from src.utils.config import AppConfig

log = structlog.get_logger()


@dataclass
class PipelineStats:
    """Statistics for pipeline performance."""
    frames_processed: int = 0
    violations_detected: int = 0
    alerts_sent: int = 0
    avg_frame_time_ms: float = 0.0
    avg_detection_time_ms: float = 0.0
    avg_chain_time_ms: float = 0.0
    fps: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    detections: DetectionResult
    violations: list[ZoneViolation]
    chain_result: Optional[ChainResult] = None
    processing_time_ms: float = 0.0


class SafetyPipelineOrchestrator:
    """
    Main orchestrator for the safety monitoring pipeline.

    Architecture:
    1. Fast path (<50ms): Camera -> YOLO -> Zone Check
    2. Slow path (async, <3s): Agent Chain -> Alert -> Database

    Design principles:
    - YOLO runs synchronously (fast path)
    - Agent chain runs async (slow path)
    - Violations trigger async agent analysis
    - Non-blocking TTS alerts
    - Graceful degradation
    """

    def __init__(
        self,
        config: AppConfig,
        cameras: Optional[list[VideoCapture]] = None,
        detector: Optional[PPEDetector] = None,
        zones: Optional[ZoneManager] = None,
        agents: Optional[SafetyAgentChain] = None,
        alerts: Optional[AlertSystem] = None,
        database: Optional[ViolationDatabase] = None
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Application configuration
            cameras: List of video capture instances
            detector: PPE detector instance
            zones: Zone manager instance
            agents: Safety agent chain
            alerts: Alert system
            database: Violation database
        """
        self.config = config
        self.cameras = cameras or []
        self.detector = detector
        self.zones = zones or ZoneManager()
        self.agents = agents
        self.alerts = alerts
        self.database = database

        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._stats = PipelineStats()
        self._start_time: Optional[float] = None
        self._frame_times: list[float] = []

        # Callbacks
        self._on_frame: Optional[Callable[[FrameResult], None]] = None
        self._on_violation: Optional[Callable[[ZoneViolation], None]] = None

        log.info("pipeline_orchestrator_initialized")

    def set_callbacks(
        self,
        on_frame: Optional[Callable[[FrameResult], None]] = None,
        on_violation: Optional[Callable[[ZoneViolation], None]] = None
    ) -> None:
        """Set callback functions for events."""
        self._on_frame = on_frame
        self._on_violation = on_violation

    async def start(self) -> None:
        """Start all processing loops."""
        if self._running:
            log.warning("pipeline_already_running")
            return

        log.info("pipeline_starting")
        self._running = True
        self._start_time = time.time()

        # Initialize database
        if self.database:
            await self.database.init()

        # Start alert system
        if self.alerts:
            self._alert_manager = AlertManager(self.alerts)
            await self._alert_manager.start()

        # Start camera processing tasks
        for camera in self.cameras:
            await camera.start()
            task = asyncio.create_task(self._process_camera(camera))
            self._tasks.append(task)

        log.info("pipeline_started", num_cameras=len(self.cameras))

    async def stop(self) -> None:
        """Graceful shutdown."""
        log.info("pipeline_stopping")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop cameras
        for camera in self.cameras:
            await camera.stop()

        # Stop alert system
        if hasattr(self, '_alert_manager'):
            await self._alert_manager.stop()

        # Close database
        if self.database:
            await self.database.close()

        log.info("pipeline_stopped")

    async def _process_camera(self, camera: VideoCapture) -> None:
        """Main processing loop for a single camera."""
        log.info("camera_processing_started", camera=camera.source_id)

        while self._running:
            try:
                # Get frame (with timeout)
                frame_packet = await camera.get_frame(timeout=1.0)
                if frame_packet is None:
                    continue

                # Process frame
                result = await self._process_frame(frame_packet)

                # Call callback
                if self._on_frame:
                    self._on_frame(result)

                # Handle violations
                if result.violations:
                    await self._handle_violations(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("frame_processing_error", error=str(e), camera=camera.source_id)
                await asyncio.sleep(0.1)

        log.info("camera_processing_stopped", camera=camera.source_id)

    async def _process_frame(self, packet: FramePacket) -> FrameResult:
        """
        Process a single frame through the pipeline.

        Fast path: YOLO + Zone Check (<50ms target)
        """
        start_time = time.time()

        # YOLO detection
        detection_start = time.time()
        detections = self.detector.detect(packet.frame)
        detections.frame_id = packet.frame_id
        detection_time = (time.time() - detection_start) * 1000

        # Zone checking
        violations = self.zones.check_frame(
            detections.detections,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp
        )

        total_time = (time.time() - start_time) * 1000

        # Update stats
        self._update_stats(total_time, detection_time)

        result = FrameResult(
            frame=packet.frame,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            detections=detections,
            violations=violations,
            processing_time_ms=total_time
        )

        return result

    async def _handle_violations(self, result: FrameResult) -> None:
        """
        Handle detected violations (slow path).

        Runs async to not block main processing.
        """
        for violation in result.violations:
            # Callback
            if self._on_violation:
                self._on_violation(violation)

            # Run agent chain (async)
            if self.agents:
                asyncio.create_task(
                    self._run_agent_chain(result, violation)
                )
            else:
                # Direct alert without agent analysis
                asyncio.create_task(
                    self._send_alert(violation)
                )

            # Store in database
            if self.database:
                asyncio.create_task(
                    self._store_violation(violation, result)
                )

    async def _run_agent_chain(
        self,
        result: FrameResult,
        violation: ZoneViolation
    ) -> None:
        """Run agent chain for detailed analysis."""
        try:
            chain_result = await self.agents.process_frame(
                result.frame,
                result.detections.detections,
                violation.zone,
                frame_id=result.frame_id,
                timestamp=result.timestamp
            )

            result.chain_result = chain_result

            # Send alert with coach feedback
            if chain_result.is_violation and chain_result.feedback:
                await self._send_alert(violation, chain_result.feedback.message)

            # Update stats
            self._stats.avg_chain_time_ms = (
                self._stats.avg_chain_time_ms * 0.9 +
                chain_result.total_time_ms * 0.1
            )

        except Exception as e:
            log.error("agent_chain_error", error=str(e))
            # Fallback to simple alert
            await self._send_alert(violation)

    async def _send_alert(
        self,
        violation: ZoneViolation,
        message: Optional[str] = None
    ) -> None:
        """Send TTS alert for violation."""
        if not self.alerts or not self.alerts.enabled:
            return

        try:
            alert = self.alerts.create_alert_from_violation(violation)
            if message:
                alert.message = message

            queued = await self.alerts.queue_alert(alert)
            if queued:
                self._stats.alerts_sent += 1

        except Exception as e:
            log.error("alert_send_error", error=str(e))

    async def _store_violation(
        self,
        violation: ZoneViolation,
        result: FrameResult
    ) -> None:
        """Store violation in database."""
        try:
            # Save frame snapshot
            frame_path = None
            # Could save frame here if needed

            # Get agent reasoning if available
            reasoning = None
            coach_msg = None
            if result.chain_result:
                reasoning = result.chain_result.decision.explanation
                if result.chain_result.feedback:
                    coach_msg = result.chain_result.feedback.message

            await self.database.insert_violation(
                violation,
                frame_path=frame_path,
                agent_reasoning=reasoning,
                coach_message=coach_msg
            )

            self._stats.violations_detected += 1

        except Exception as e:
            log.error("database_store_error", error=str(e))

    def _update_stats(self, frame_time: float, detection_time: float) -> None:
        """Update pipeline statistics."""
        self._stats.frames_processed += 1

        # Rolling average for frame time
        self._frame_times.append(frame_time)
        if len(self._frame_times) > 100:
            self._frame_times = self._frame_times[-100:]

        self._stats.avg_frame_time_ms = sum(self._frame_times) / len(self._frame_times)
        self._stats.avg_detection_time_ms = (
            self._stats.avg_detection_time_ms * 0.9 + detection_time * 0.1
        )

        # Calculate FPS
        if self._start_time:
            elapsed = time.time() - self._start_time
            self._stats.uptime_seconds = elapsed
            if elapsed > 0:
                self._stats.fps = self._stats.frames_processed / elapsed

    def get_status(self) -> dict:
        """Get real-time pipeline status."""
        camera_stats = {}
        for cam in self.cameras:
            camera_stats[cam.source_id] = cam.get_stats()

        return {
            "running": self._running,
            "stats": {
                "frames_processed": self._stats.frames_processed,
                "violations_detected": self._stats.violations_detected,
                "alerts_sent": self._stats.alerts_sent,
                "avg_frame_time_ms": round(self._stats.avg_frame_time_ms, 1),
                "avg_detection_time_ms": round(self._stats.avg_detection_time_ms, 1),
                "avg_chain_time_ms": round(self._stats.avg_chain_time_ms, 1),
                "fps": round(self._stats.fps, 1),
                "uptime_seconds": round(self._stats.uptime_seconds, 0)
            },
            "cameras": camera_stats,
            "zones": len(self.zones.list_zones()),
            "agents": self.agents.get_stats() if self.agents else None
        }

    def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest frame from a camera."""
        for cam in self.cameras:
            if cam.source_id == camera_id:
                packet = cam.get_frame_nowait()
                return packet.frame if packet else None
        return None


async def create_pipeline(config: AppConfig) -> SafetyPipelineOrchestrator:
    """
    Create a fully configured pipeline from config.

    Args:
        config: Application configuration

    Returns:
        Configured SafetyPipelineOrchestrator
    """
    # Create cameras
    cameras = []
    for cam_config in config.cameras:
        if cam_config.enabled:
            camera = VideoCapture(
                source=cam_config.source,
                fps_target=cam_config.fps,
                source_id=cam_config.name,
                buffer_size=cam_config.buffer_size
            )
            cameras.append(camera)

    # Create detector
    detector = PPEDetector(
        model_size=config.detection.model,
        confidence_threshold=config.detection.confidence,
        device=config.detection.device,
        custom_classes=config.detection.classes if config.detection.classes else None,
        enable_tracking=config.detection.enable_tracking
    )

    # Create zone manager
    zones = ZoneManager(config.zones_config)

    # Create agent chain
    from src.agents.perception_agent import PerceptionAgent
    from src.agents.policy_agent import PolicyAgent
    from src.agents.coach_agent import CoachAgent

    # Use mock VLM for now (no GPU required)
    perception = PerceptionAgent(use_mock=not config.vlm.enabled)
    policy = PolicyAgent()
    coach = CoachAgent()

    agents = SafetyAgentChain(
        perception=perception,
        policy=policy,
        coach=coach,
        max_chain_time_ms=config.agents.max_analysis_time_ms
    )

    # Create alert system
    alerts = AlertSystem(
        engine=config.alerts.engine,
        voice=config.alerts.voice,
        cooldown_seconds=config.alerts.cooldown_seconds,
        enabled=config.alerts.enabled,
        cache_dir=config.alerts.cache_dir
    )

    # Create database
    database = ViolationDatabase(config.database_path)

    return SafetyPipelineOrchestrator(
        config=config,
        cameras=cameras,
        detector=detector,
        zones=zones,
        agents=agents,
        alerts=alerts,
        database=database
    )
