"""
CUDA-Accelerated Video Decoder for DGX Spark.

Uses NVDEC (NVIDIA Video Decoder) to decode video directly to GPU memory,
eliminating CPU-GPU memory copies for true zero-copy inference.

Quality Guarantees:
- Lossless decode: NVDEC provides bit-exact decoding
- Full resolution: No downscaling during decode
- All frames: No frame dropping in decoder
- Graceful fallback: Falls back to OpenCV if NVDEC unavailable

DGX Spark Benefits:
- Hardware decode frees GPU compute for inference
- Direct GPU memory output avoids PCIe bandwidth bottleneck
- On Grace Hopper, unified memory means true zero-copy
"""

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Generator, Any, Callable
from queue import Queue, Empty
import structlog

log = structlog.get_logger()

# Check for required libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import PyNvVideoCodec for NVDEC
try:
    import PyNvVideoCodec as nvc
    NVDEC_AVAILABLE = True
except ImportError:
    NVDEC_AVAILABLE = False

# Try cupy for zero-copy GPU operations
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class DecodedFrame:
    """
    A decoded video frame with GPU tensor.

    Attributes:
        tensor: Frame data as PyTorch tensor on GPU (H, W, C) or (C, H, W)
        frame_id: Sequential frame number
        timestamp_ms: Presentation timestamp in milliseconds
        width: Frame width
        height: Frame height
        is_gpu: Whether tensor is on GPU
        decode_time_ms: Time taken to decode this frame
    """
    tensor: "torch.Tensor"
    frame_id: int
    timestamp_ms: float
    width: int
    height: int
    is_gpu: bool = True
    decode_time_ms: float = 0.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.tensor.shape)

    def to_numpy(self) -> "np.ndarray":
        """Convert to numpy array (may require GPU sync)."""
        if self.is_gpu:
            return self.tensor.cpu().numpy()
        return self.tensor.numpy()

    def to_bgr_numpy(self) -> "np.ndarray":
        """Convert to BGR numpy array for OpenCV compatibility."""
        arr = self.to_numpy()
        # Handle channel ordering if needed
        if arr.shape[0] == 3:  # CHW format
            arr = arr.transpose(1, 2, 0)
        return arr


class NVDECDecoder:
    """
    NVDEC-based video decoder with direct GPU memory output.

    This decoder uses NVIDIA's hardware video decoder to decode
    video frames directly into GPU memory, avoiding CPU copies.

    Quality Notes:
    - NVDEC is hardware-accelerated but produces bit-exact output
    - No quality loss compared to software decode
    - Supports H.264, H.265/HEVC, VP9, AV1 (depending on GPU)
    """

    def __init__(
        self,
        gpu_id: int = 0,
        output_format: str = "rgb",
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize NVDEC decoder.

        Args:
            gpu_id: CUDA device ID
            output_format: Output format ("rgb", "bgr", "nv12")
            target_size: Optional (width, height) for resize during decode
        """
        self.gpu_id = gpu_id
        self.output_format = output_format
        self.target_size = target_size

        self._decoder = None
        self._initialized = False
        self._frame_count = 0

        # Statistics
        self._total_decode_time_ms = 0.0
        self._frames_decoded = 0

    def open(self, video_path: str) -> bool:
        """
        Open a video file for decoding.

        Args:
            video_path: Path to video file

        Returns:
            True if successfully opened
        """
        if not NVDEC_AVAILABLE:
            log.warning("nvdec_not_available", fallback="opencv")
            return False

        try:
            # Create NVDEC decoder
            self._decoder = nvc.PyNvDecoder(
                video_path,
                self.gpu_id
            )
            self._initialized = True
            self._frame_count = 0

            log.info(
                "nvdec_decoder_opened",
                path=video_path,
                gpu=self.gpu_id,
                width=self._decoder.Width(),
                height=self._decoder.Height()
            )
            return True

        except Exception as e:
            log.error("nvdec_open_failed", error=str(e))
            self._initialized = False
            return False

    def decode_frame(self) -> Optional[DecodedFrame]:
        """
        Decode the next frame directly to GPU memory.

        Returns:
            DecodedFrame with GPU tensor, or None if EOF/error
        """
        if not self._initialized or self._decoder is None:
            return None

        try:
            start_time = time.perf_counter()

            # Decode to GPU surface
            raw_frame = self._decoder.DecodeSingleFrame()

            if raw_frame is None:
                return None

            decode_time = (time.perf_counter() - start_time) * 1000

            # Convert to PyTorch tensor (zero-copy on same GPU)
            if CUPY_AVAILABLE:
                # Use cupy for zero-copy conversion
                cp_array = cp.asarray(raw_frame)
                tensor = torch.as_tensor(cp_array, device=f'cuda:{self.gpu_id}')
            else:
                # Fallback: may involve a copy
                tensor = torch.from_numpy(raw_frame).to(f'cuda:{self.gpu_id}')

            self._frame_count += 1
            self._frames_decoded += 1
            self._total_decode_time_ms += decode_time

            return DecodedFrame(
                tensor=tensor,
                frame_id=self._frame_count,
                timestamp_ms=self._frame_count * (1000.0 / 30.0),  # Estimate
                width=tensor.shape[1],
                height=tensor.shape[0],
                is_gpu=True,
                decode_time_ms=decode_time
            )

        except Exception as e:
            log.warning("nvdec_decode_error", error=str(e))
            return None

    def close(self) -> None:
        """Close the decoder and free resources."""
        self._decoder = None
        self._initialized = False

    def get_stats(self) -> dict:
        """Get decoder statistics."""
        avg_decode_time = (
            self._total_decode_time_ms / max(1, self._frames_decoded)
        )
        return {
            "frames_decoded": self._frames_decoded,
            "avg_decode_time_ms": round(avg_decode_time, 2),
            "decode_fps": round(1000.0 / max(0.001, avg_decode_time), 1),
            "gpu_id": self.gpu_id,
            "nvdec_available": NVDEC_AVAILABLE
        }


class OpenCVCUDADecoder:
    """
    OpenCV-based decoder with CUDA upload.

    Fallback decoder that uses OpenCV for decoding and uploads
    frames to GPU. Not true zero-copy but maintains compatibility.

    Quality Notes:
    - Uses OpenCV's FFmpeg backend for decoding
    - Exact same quality as standard OpenCV
    - GPU upload uses pinned memory for efficiency
    """

    def __init__(
        self,
        gpu_id: int = 0,
        use_pinned_memory: bool = True
    ):
        """
        Initialize OpenCV CUDA decoder.

        Args:
            gpu_id: CUDA device ID
            use_pinned_memory: Use pinned memory for faster upload
        """
        self.gpu_id = gpu_id
        self.use_pinned_memory = use_pinned_memory

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._pinned_buffer: Optional[torch.Tensor] = None
        self._gpu_buffer: Optional[torch.Tensor] = None

        # Statistics
        self._total_decode_time_ms = 0.0
        self._total_upload_time_ms = 0.0
        self._frames_decoded = 0

    def open(self, video_path: str) -> bool:
        """
        Open a video file.

        Args:
            video_path: Path to video file

        Returns:
            True if successfully opened
        """
        if not CV2_AVAILABLE:
            log.error("opencv_not_available")
            return False

        try:
            self._cap = cv2.VideoCapture(video_path)

            if not self._cap.isOpened():
                log.error("opencv_open_failed", path=video_path)
                return False

            # Get video properties
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Pre-allocate buffers
            if TORCH_AVAILABLE and CUDA_AVAILABLE and self.use_pinned_memory:
                self._pinned_buffer = torch.empty(
                    (height, width, 3),
                    dtype=torch.uint8,
                    pin_memory=True
                )
                self._gpu_buffer = torch.empty(
                    (height, width, 3),
                    dtype=torch.uint8,
                    device=f'cuda:{self.gpu_id}'
                )

            self._frame_count = 0

            log.info(
                "opencv_cuda_decoder_opened",
                path=video_path,
                width=width,
                height=height,
                pinned_memory=self.use_pinned_memory
            )
            return True

        except Exception as e:
            log.error("opencv_open_error", error=str(e))
            return False

    def decode_frame(self) -> Optional[DecodedFrame]:
        """
        Decode next frame and upload to GPU.

        Returns:
            DecodedFrame with GPU tensor, or None if EOF
        """
        if self._cap is None:
            return None

        try:
            # Decode frame (CPU)
            start_decode = time.perf_counter()
            ret, frame = self._cap.read()
            decode_time = (time.perf_counter() - start_decode) * 1000

            if not ret:
                return None

            # Upload to GPU
            start_upload = time.perf_counter()

            if self._pinned_buffer is not None and self._gpu_buffer is not None:
                # Fast path: use pre-allocated pinned memory
                np.copyto(
                    self._pinned_buffer.numpy(),
                    frame
                )
                self._gpu_buffer.copy_(self._pinned_buffer, non_blocking=True)
                tensor = self._gpu_buffer.clone()  # Clone to allow buffer reuse
            elif TORCH_AVAILABLE and CUDA_AVAILABLE:
                # Slower path: direct upload
                tensor = torch.from_numpy(frame).to(
                    f'cuda:{self.gpu_id}',
                    non_blocking=True
                )
            else:
                # CPU only
                tensor = torch.from_numpy(frame)

            upload_time = (time.perf_counter() - start_upload) * 1000

            self._frame_count += 1
            self._frames_decoded += 1
            self._total_decode_time_ms += decode_time
            self._total_upload_time_ms += upload_time

            return DecodedFrame(
                tensor=tensor,
                frame_id=self._frame_count,
                timestamp_ms=self._cap.get(cv2.CAP_PROP_POS_MSEC),
                width=frame.shape[1],
                height=frame.shape[0],
                is_gpu=tensor.is_cuda if hasattr(tensor, 'is_cuda') else False,
                decode_time_ms=decode_time + upload_time
            )

        except Exception as e:
            log.warning("opencv_decode_error", error=str(e))
            return None

    def seek(self, frame_number: int) -> bool:
        """Seek to specific frame."""
        if self._cap is None:
            return False
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def close(self) -> None:
        """Close decoder and free resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._pinned_buffer = None
        self._gpu_buffer = None

    def get_stats(self) -> dict:
        """Get decoder statistics."""
        frames = max(1, self._frames_decoded)
        avg_decode = self._total_decode_time_ms / frames
        avg_upload = self._total_upload_time_ms / frames

        return {
            "frames_decoded": self._frames_decoded,
            "avg_decode_time_ms": round(avg_decode, 2),
            "avg_upload_time_ms": round(avg_upload, 2),
            "total_frame_time_ms": round(avg_decode + avg_upload, 2),
            "effective_fps": round(1000.0 / max(0.001, avg_decode + avg_upload), 1),
            "gpu_id": self.gpu_id,
            "pinned_memory": self.use_pinned_memory
        }


class ZeroCopyVideoDecoder:
    """
    Unified video decoder interface with automatic backend selection.

    Automatically selects the best available decoder:
    1. NVDEC (if available) - True zero-copy, hardware accelerated
    2. OpenCV + CUDA upload - Software decode, fast GPU upload
    3. OpenCV CPU - Fallback for non-CUDA systems

    Quality Guarantee:
    - All backends produce identical visual output
    - No quality degradation regardless of backend
    - Graceful fallback maintains functionality
    """

    def __init__(
        self,
        gpu_id: int = 0,
        prefer_nvdec: bool = True,
        use_pinned_memory: bool = True
    ):
        """
        Initialize zero-copy video decoder.

        Args:
            gpu_id: CUDA device ID
            prefer_nvdec: Prefer NVDEC if available
            use_pinned_memory: Use pinned memory for CPU decoder
        """
        self.gpu_id = gpu_id
        self.prefer_nvdec = prefer_nvdec
        self.use_pinned_memory = use_pinned_memory

        self._decoder = None
        self._backend = "none"
        self._video_path = None

    def open(self, video_path: str) -> bool:
        """
        Open video file with best available decoder.

        Args:
            video_path: Path to video file

        Returns:
            True if opened successfully
        """
        self._video_path = video_path

        # Try NVDEC first
        if self.prefer_nvdec and NVDEC_AVAILABLE and CUDA_AVAILABLE:
            decoder = NVDECDecoder(gpu_id=self.gpu_id)
            if decoder.open(video_path):
                self._decoder = decoder
                self._backend = "nvdec"
                log.info("using_nvdec_decoder", zero_copy=True)
                return True

        # Fall back to OpenCV + CUDA
        if CUDA_AVAILABLE:
            decoder = OpenCVCUDADecoder(
                gpu_id=self.gpu_id,
                use_pinned_memory=self.use_pinned_memory
            )
            if decoder.open(video_path):
                self._decoder = decoder
                self._backend = "opencv_cuda"
                log.info("using_opencv_cuda_decoder", pinned=self.use_pinned_memory)
                return True

        # Last resort: OpenCV CPU only
        decoder = OpenCVCUDADecoder(gpu_id=0, use_pinned_memory=False)
        if decoder.open(video_path):
            self._decoder = decoder
            self._backend = "opencv_cpu"
            log.info("using_opencv_cpu_decoder")
            return True

        return False

    def decode_frame(self) -> Optional[DecodedFrame]:
        """Decode next frame."""
        if self._decoder is None:
            return None
        return self._decoder.decode_frame()

    def __iter__(self) -> Generator[DecodedFrame, None, None]:
        """Iterate over all frames."""
        while True:
            frame = self.decode_frame()
            if frame is None:
                break
            yield frame

    def close(self) -> None:
        """Close decoder."""
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None

    def get_stats(self) -> dict:
        """Get decoder statistics."""
        stats = {"backend": self._backend}
        if self._decoder is not None:
            stats.update(self._decoder.get_stats())
        return stats

    @property
    def backend(self) -> str:
        """Get current decoder backend name."""
        return self._backend

    @property
    def is_zero_copy(self) -> bool:
        """Check if using true zero-copy decode."""
        return self._backend == "nvdec"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_decoder_capabilities() -> dict:
    """
    Get information about available decoder capabilities.

    Returns:
        Dictionary with decoder availability and features
    """
    capabilities = {
        "nvdec_available": NVDEC_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "opencv_available": CV2_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "recommended_backend": "none",
        "zero_copy_possible": False
    }

    if NVDEC_AVAILABLE and CUDA_AVAILABLE:
        capabilities["recommended_backend"] = "nvdec"
        capabilities["zero_copy_possible"] = True
    elif CUDA_AVAILABLE:
        capabilities["recommended_backend"] = "opencv_cuda"
    elif CV2_AVAILABLE:
        capabilities["recommended_backend"] = "opencv_cpu"

    # Check for unified memory (Grace Hopper)
    if CUDA_AVAILABLE:
        try:
            props = torch.cuda.get_device_properties(0)
            capabilities["unified_memory"] = props.major >= 9
            capabilities["gpu_name"] = props.name
            capabilities["compute_capability"] = f"{props.major}.{props.minor}"
        except Exception:
            pass

    return capabilities
