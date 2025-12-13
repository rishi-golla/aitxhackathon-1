"""
Video Capture Pipeline for Factory Safety System.

Production-ready async video capture with support for multiple sources.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional, Union
import cv2
import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class FramePacket:
    """Container for captured video frames with metadata."""
    frame: np.ndarray  # BGR format
    frame_id: int
    timestamp: float  # time.time()
    source_id: str  # camera identifier
    resolution: tuple[int, int]  # (width, height)


@dataclass
class CaptureStats:
    """Statistics for video capture performance."""
    fps_actual: float = 0.0
    fps_target: float = 30.0
    dropped_frames: int = 0
    total_frames: int = 0
    latency_ms: float = 0.0
    reconnect_count: int = 0
    is_connected: bool = False


class VideoCapture:
    """
    Async video capture pipeline supporting webcams, RTSP streams, and video files.

    Features:
    - Background capture thread with async interface
    - Frame skipping for real-time processing
    - Graceful reconnection for streams
    - Optional resolution scaling
    - CUDA-accelerated decode when available
    """

    def __init__(
        self,
        source: Union[str, int],
        fps_target: int = 30,
        buffer_size: int = 10,
        source_id: Optional[str] = None,
        scale_resolution: Optional[tuple[int, int]] = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0
    ):
        """
        Initialize video capture.

        Args:
            source: RTSP URL, webcam index, or video file path
            fps_target: Target FPS (skip frames if source is faster)
            buffer_size: Frame buffer size for async processing
            source_id: Identifier for this camera
            scale_resolution: Optional (width, height) to scale frames
            reconnect_attempts: Number of reconnection attempts for streams
            reconnect_delay: Initial delay between reconnect attempts (exponential backoff)
        """
        self.source = source
        self.fps_target = fps_target
        self.buffer_size = buffer_size
        self.source_id = source_id or str(source)
        self.scale_resolution = scale_resolution
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._buffer: Queue[FramePacket] = Queue(maxsize=buffer_size)
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_id = 0
        self._stats = CaptureStats(fps_target=fps_target)
        self._last_frame_time = 0.0
        self._fps_window: list[float] = []
        self._lock = threading.Lock()

        # Check for CUDA decode support
        self._use_cuda = self._check_cuda_support()

    def _check_cuda_support(self) -> bool:
        """Check if CUDA video decoding is available."""
        try:
            # Check if OpenCV was built with CUDA support
            build_info = cv2.getBuildInformation()
            return "CUDA" in build_info and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """Create video capture with optimal backend."""
        cap = None

        try:
            if isinstance(self.source, int):
                # Webcam - try different backends
                for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                    cap = cv2.VideoCapture(self.source, backend)
                    if cap.isOpened():
                        break
            elif str(self.source).startswith(("rtsp://", "rtmp://", "http://")):
                # Network stream
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    # Optimize for streaming
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                # Video file
                cap = cv2.VideoCapture(self.source)

            if cap and cap.isOpened():
                # Set capture properties
                if self.scale_resolution:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.scale_resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.scale_resolution[1])

                log.info(
                    "video_capture_opened",
                    source=self.source,
                    source_id=self.source_id,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=cap.get(cv2.CAP_PROP_FPS)
                )
                return cap
            else:
                log.error("video_capture_failed", source=self.source)
                return None

        except Exception as e:
            log.error("video_capture_error", source=self.source, error=str(e))
            if cap:
                cap.release()
            return None

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the video source."""
        delay = self.reconnect_delay

        for attempt in range(self.reconnect_attempts):
            log.warning(
                "video_capture_reconnecting",
                source_id=self.source_id,
                attempt=attempt + 1,
                max_attempts=self.reconnect_attempts
            )

            time.sleep(delay)
            delay *= 2  # Exponential backoff

            if self._capture:
                self._capture.release()

            self._capture = self._create_capture()
            if self._capture and self._capture.isOpened():
                with self._lock:
                    self._stats.reconnect_count += 1
                    self._stats.is_connected = True
                log.info("video_capture_reconnected", source_id=self.source_id)
                return True

        log.error(
            "video_capture_reconnect_failed",
            source_id=self.source_id,
            attempts=self.reconnect_attempts
        )
        return False

    def _capture_loop(self) -> None:
        """Background capture loop running in separate thread."""
        frame_interval = 1.0 / self.fps_target
        is_file = isinstance(self.source, str) and not self.source.startswith(("rtsp://", "rtmp://", "http://"))
        no_frame_count = 0
        max_no_frames = 300  # ~10 seconds at 30fps

        while self._running:
            if not self._capture or not self._capture.isOpened():
                with self._lock:
                    self._stats.is_connected = False

                if not self._reconnect():
                    break
                continue

            start_time = time.time()

            # Read frame
            success, frame = self._capture.read()

            if not success:
                no_frame_count += 1

                if is_file:
                    # Loop video file
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    no_frame_count = 0
                    continue
                elif no_frame_count > max_no_frames:
                    log.warning("video_capture_timeout", source_id=self.source_id)
                    with self._lock:
                        self._stats.is_connected = False
                    if not self._reconnect():
                        break
                    no_frame_count = 0
                continue

            no_frame_count = 0

            # Scale frame if needed
            if self.scale_resolution and frame.shape[1::-1] != self.scale_resolution:
                frame = cv2.resize(frame, self.scale_resolution, interpolation=cv2.INTER_LINEAR)

            # Create frame packet
            self._frame_id += 1
            packet = FramePacket(
                frame=frame,
                frame_id=self._frame_id,
                timestamp=time.time(),
                source_id=self.source_id,
                resolution=(frame.shape[1], frame.shape[0])
            )

            # Update stats
            capture_time = time.time() - start_time
            with self._lock:
                self._stats.total_frames += 1
                self._stats.latency_ms = capture_time * 1000
                self._stats.is_connected = True

                # Calculate FPS
                now = time.time()
                self._fps_window.append(now)
                self._fps_window = [t for t in self._fps_window if now - t < 1.0]
                self._stats.fps_actual = len(self._fps_window)

            # Add to buffer (drop oldest if full)
            try:
                if self._buffer.full():
                    try:
                        self._buffer.get_nowait()
                        with self._lock:
                            self._stats.dropped_frames += 1
                    except Empty:
                        pass
                self._buffer.put_nowait(packet)
            except Exception as e:
                log.warning("buffer_error", error=str(e))

            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    async def start(self) -> None:
        """Start the capture pipeline."""
        if self._running:
            log.warning("video_capture_already_running", source_id=self.source_id)
            return

        self._capture = self._create_capture()
        if not self._capture:
            raise ConnectionError(f"Could not open video source: {self.source}")

        self._running = True
        self._stats.is_connected = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        log.info("video_capture_started", source_id=self.source_id)

    async def stop(self) -> None:
        """Stop the capture pipeline."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._capture:
            self._capture.release()
            self._capture = None

        # Clear buffer
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except Empty:
                break

        with self._lock:
            self._stats.is_connected = False

        log.info("video_capture_stopped", source_id=self.source_id)

    async def get_frame(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """
        Get the next frame from the buffer.

        Args:
            timeout: Maximum time to wait for a frame

        Returns:
            FramePacket or None if no frame available
        """
        try:
            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._buffer.get(timeout=timeout)
            )
        except Empty:
            return None

    def get_frame_nowait(self) -> Optional[FramePacket]:
        """Get a frame without waiting."""
        try:
            return self._buffer.get_nowait()
        except Empty:
            return None

    def get_stats(self) -> dict:
        """Get capture statistics."""
        with self._lock:
            return {
                "source_id": self.source_id,
                "fps_actual": round(self._stats.fps_actual, 1),
                "fps_target": self._stats.fps_target,
                "dropped_frames": self._stats.dropped_frames,
                "total_frames": self._stats.total_frames,
                "latency_ms": round(self._stats.latency_ms, 2),
                "reconnect_count": self._stats.reconnect_count,
                "is_connected": self._stats.is_connected,
                "buffer_size": self._buffer.qsize(),
                "buffer_capacity": self.buffer_size
            }

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if connected to source."""
        with self._lock:
            return self._stats.is_connected


class MultiCameraCapture:
    """Manager for multiple camera captures."""

    def __init__(self):
        self.cameras: dict[str, VideoCapture] = {}

    def add_camera(
        self,
        source: Union[str, int],
        source_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Add a camera to the manager."""
        camera = VideoCapture(source, source_id=source_id, **kwargs)
        self.cameras[camera.source_id] = camera
        return camera.source_id

    async def start_all(self) -> None:
        """Start all cameras concurrently."""
        await asyncio.gather(*[cam.start() for cam in self.cameras.values()])

    async def stop_all(self) -> None:
        """Stop all cameras."""
        await asyncio.gather(*[cam.stop() for cam in self.cameras.values()])

    async def get_frames(self) -> dict[str, Optional[FramePacket]]:
        """Get frames from all cameras."""
        results = {}
        for source_id, camera in self.cameras.items():
            results[source_id] = camera.get_frame_nowait()
        return results

    def get_all_stats(self) -> dict[str, dict]:
        """Get stats for all cameras."""
        return {source_id: cam.get_stats() for source_id, cam in self.cameras.items()}
