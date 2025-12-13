"""
Text-to-Speech Alert System for safety violations.

Provides voice alerts for factory safety violations with priority queuing.
"""

import asyncio
import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
import structlog

log = structlog.get_logger()


class AlertPriority(Enum):
    """Priority levels for alerts."""
    INFO = 1  # General reminder
    WARNING = 2  # Compliance issue
    CRITICAL = 3  # Immediate hazard


@dataclass
class Alert:
    """A safety alert to be spoken."""
    alert_id: str
    message: str
    priority: AlertPriority
    zone_id: Optional[str] = None
    worker_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    spoken: bool = False
    audio_path: Optional[str] = None
    violation_type: Optional[str] = None


# Alert message templates
ALERT_TEMPLATES = {
    "missing_ppe": "Attention in {zone}. {item} required. Please equip before continuing.",
    "missing_ppe_multiple": "Attention in {zone}. Missing {items}. Please equip required safety gear.",
    "restricted_area": "Warning. You are entering a restricted area. Authorization required.",
    "overcrowding": "Zone {zone} at capacity. Please wait for clearance.",
    "hazard_proximity": "Caution. Active hazard detected nearby. Maintain safe distance.",
    "general_reminder": "Safety reminder for {zone}. {message}",
    "machine_zone": "Warning. You are near active machinery. Keep hands clear and wear required PPE.",
}


class AlertSystem:
    """
    Text-to-speech alert system with priority queuing.

    Features:
    - Multiple TTS engine support (piper, edge-tts, gtts)
    - Priority queue (critical alerts interrupt lower priority)
    - Rate limiting per zone
    - Audio caching
    - Async non-blocking playback
    """

    def __init__(
        self,
        engine: str = "gtts",  # "piper", "edge", "gtts"
        voice: str = "en",
        rate: float = 1.0,
        cache_dir: str = ".alert_cache",
        cooldown_seconds: int = 30,
        enabled: bool = True
    ):
        """
        Initialize alert system.

        Args:
            engine: TTS engine to use ("piper", "edge", "gtts")
            voice: Voice identifier
            rate: Speech rate multiplier
            cache_dir: Directory for cached audio files
            cooldown_seconds: Minimum time between alerts for same zone/worker
            enabled: Whether alerts are enabled
        """
        self.engine = engine
        self.voice = voice
        self.rate = rate
        self.cache_dir = Path(cache_dir)
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled

        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._processing = False
        self._last_alert_time: dict[str, float] = {}  # zone_id/worker_id -> timestamp
        self._pending_alerts: list[Alert] = []
        self._audio_player = None
        self._tts_engine = None

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TTS engine
        self._init_tts_engine()

        log.info(
            "alert_system_init",
            engine=engine,
            voice=voice,
            cache_dir=str(self.cache_dir)
        )

    def _init_tts_engine(self) -> None:
        """Initialize the TTS engine."""
        if self.engine == "piper":
            try:
                from piper import PiperVoice
                self._tts_engine = "piper"
                log.info("tts_engine_piper_ready")
            except ImportError:
                log.warning("piper_not_available_fallback_gtts")
                self.engine = "gtts"

        if self.engine == "edge":
            try:
                import edge_tts
                self._tts_engine = "edge"
                log.info("tts_engine_edge_ready")
            except ImportError:
                log.warning("edge_tts_not_available_fallback_gtts")
                self.engine = "gtts"

        if self.engine == "gtts" or self._tts_engine is None:
            try:
                from gtts import gTTS
                self._tts_engine = "gtts"
                log.info("tts_engine_gtts_ready")
            except ImportError:
                log.error("no_tts_engine_available")
                self._tts_engine = None

    def _get_cache_path(self, message: str) -> Path:
        """Get cache path for a message."""
        message_hash = hashlib.md5(message.encode()).hexdigest()[:16]
        return self.cache_dir / f"{message_hash}.mp3"

    async def generate_alert(self, alert: Alert) -> Optional[str]:
        """
        Generate audio file for alert.

        Args:
            alert: Alert to generate audio for

        Returns:
            Path to audio file, or None if generation failed
        """
        if not self._tts_engine:
            log.warning("no_tts_engine_cannot_generate")
            return None

        # Check cache
        cache_path = self._get_cache_path(alert.message)
        if cache_path.exists():
            log.debug("alert_audio_cached", path=str(cache_path))
            return str(cache_path)

        try:
            if self._tts_engine == "gtts":
                from gtts import gTTS
                tts = gTTS(text=alert.message, lang="en", slow=False)
                tts.save(str(cache_path))

            elif self._tts_engine == "edge":
                import edge_tts
                communicate = edge_tts.Communicate(
                    alert.message,
                    voice=self.voice or "en-US-AriaNeural"
                )
                await communicate.save(str(cache_path))

            elif self._tts_engine == "piper":
                from piper import PiperVoice
                voice = PiperVoice.load(self.voice)
                audio = voice.synthesize(alert.message)
                with open(cache_path, "wb") as f:
                    f.write(audio)

            alert.audio_path = str(cache_path)
            log.info("alert_audio_generated", path=str(cache_path))
            return str(cache_path)

        except Exception as e:
            log.error("alert_audio_generation_failed", error=str(e))
            return None

    async def _play_audio(self, audio_path: str) -> None:
        """Play audio file."""
        try:
            # Try pygame first
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()

                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()

                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                return
            except ImportError:
                pass

            # Try sounddevice + soundfile
            try:
                import sounddevice as sd
                import soundfile as sf

                data, samplerate = sf.read(audio_path)
                sd.play(data, samplerate)
                sd.wait()
                return
            except ImportError:
                pass

            # Fallback to system command
            import subprocess
            import platform

            system = platform.system()
            if system == "Darwin":
                subprocess.run(["afplay", audio_path], check=True)
            elif system == "Linux":
                subprocess.run(["aplay", audio_path], check=True)
            elif system == "Windows":
                import winsound
                winsound.PlaySound(audio_path, winsound.SND_FILENAME)

        except Exception as e:
            log.error("audio_playback_failed", error=str(e), path=audio_path)

    async def speak(self, alert: Alert) -> None:
        """
        Play alert audio (non-blocking).

        Args:
            alert: Alert to speak
        """
        if not self.enabled:
            log.debug("alerts_disabled_not_speaking")
            return

        # Generate audio if needed
        if not alert.audio_path:
            audio_path = await self.generate_alert(alert)
            if not audio_path:
                return
            alert.audio_path = audio_path

        # Play audio
        await self._play_audio(alert.audio_path)
        alert.spoken = True
        log.info("alert_spoken", alert_id=alert.alert_id, message=alert.message)

    def _check_cooldown(self, alert: Alert) -> bool:
        """Check if alert should be throttled due to cooldown."""
        # Create cooldown key
        if alert.worker_id:
            cooldown_key = f"worker_{alert.worker_id}_{alert.violation_type}"
        elif alert.zone_id:
            cooldown_key = f"zone_{alert.zone_id}_{alert.violation_type}"
        else:
            cooldown_key = f"general_{alert.violation_type}"

        now = time.time()
        last_time = self._last_alert_time.get(cooldown_key, 0)

        if now - last_time < self.cooldown_seconds:
            log.debug(
                "alert_throttled",
                key=cooldown_key,
                time_since_last=now - last_time
            )
            return False

        self._last_alert_time[cooldown_key] = now
        return True

    async def queue_alert(self, alert: Alert) -> bool:
        """
        Add alert to priority queue.

        Args:
            alert: Alert to queue

        Returns:
            True if queued, False if throttled
        """
        if not self.enabled:
            return False

        if not self._check_cooldown(alert):
            return False

        # Priority queue uses (priority, timestamp, alert) tuples
        # Lower priority value = higher priority
        priority = -alert.priority.value  # Negate so CRITICAL (3) becomes -3 (highest)

        await self._queue.put((priority, alert.timestamp, alert))
        self._pending_alerts.append(alert)

        log.info(
            "alert_queued",
            alert_id=alert.alert_id,
            priority=alert.priority.name,
            zone=alert.zone_id
        )
        return True

    async def process_queue(self) -> None:
        """Background worker for alert playback."""
        self._processing = True

        while self._processing:
            try:
                # Get next alert (with timeout to allow shutdown)
                try:
                    priority, timestamp, alert = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Speak alert
                await self.speak(alert)

                # Remove from pending
                if alert in self._pending_alerts:
                    self._pending_alerts.remove(alert)

                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("queue_processing_error", error=str(e))

        log.info("alert_queue_processor_stopped")

    async def stop_processing(self) -> None:
        """Stop queue processing."""
        self._processing = False

    def get_pending_alerts(self) -> list[Alert]:
        """Get list of pending alerts."""
        return self._pending_alerts.copy()

    def clear_queue(self) -> None:
        """Clear all pending alerts."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._pending_alerts.clear()
        log.info("alert_queue_cleared")

    def create_alert_from_violation(
        self,
        violation,  # ZoneViolation
        priority_override: Optional[AlertPriority] = None
    ) -> Alert:
        """
        Create an alert from a zone violation.

        Args:
            violation: ZoneViolation object
            priority_override: Override default priority

        Returns:
            Alert object ready to be queued
        """
        zone_name = violation.zone.name

        # Determine message based on violation type
        if violation.violation_type == "missing_ppe":
            if len(violation.missing_items) == 1:
                message = ALERT_TEMPLATES["missing_ppe"].format(
                    zone=zone_name,
                    item=violation.missing_items[0]
                )
            else:
                items_str = ", ".join(violation.missing_items[:-1])
                items_str += f" and {violation.missing_items[-1]}"
                message = ALERT_TEMPLATES["missing_ppe_multiple"].format(
                    zone=zone_name,
                    items=items_str
                )
            default_priority = AlertPriority.WARNING

        elif violation.violation_type == "restricted_area":
            message = ALERT_TEMPLATES["restricted_area"]
            default_priority = AlertPriority.CRITICAL

        elif violation.violation_type == "overcrowded":
            message = ALERT_TEMPLATES["overcrowding"].format(zone=zone_name)
            default_priority = AlertPriority.WARNING

        else:
            message = f"Safety alert in {zone_name}. Please check compliance."
            default_priority = AlertPriority.INFO

        return Alert(
            alert_id=str(uuid.uuid4()),
            message=message,
            priority=priority_override or default_priority,
            zone_id=violation.zone.zone_id,
            worker_id=violation.worker_id,
            timestamp=violation.timestamp,
            violation_type=violation.violation_type
        )


class AlertManager:
    """High-level manager for the alert system."""

    def __init__(self, alert_system: AlertSystem):
        self.alert_system = alert_system
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the alert manager."""
        self._task = asyncio.create_task(self.alert_system.process_queue())
        log.info("alert_manager_started")

    async def stop(self) -> None:
        """Stop the alert manager."""
        await self.alert_system.stop_processing()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("alert_manager_stopped")

    async def handle_violation(self, violation) -> None:
        """Handle a zone violation by creating and queuing an alert."""
        alert = self.alert_system.create_alert_from_violation(violation)
        await self.alert_system.queue_alert(alert)

    async def handle_violations(self, violations: list) -> None:
        """Handle multiple violations."""
        for violation in violations:
            await self.handle_violation(violation)
