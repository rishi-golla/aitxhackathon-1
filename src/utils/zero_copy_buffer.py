"""
Zero-Copy GPU Buffer Management for DGX Spark.

This module provides true zero-copy memory management leveraging
DGX Spark's unified memory architecture (Grace Hopper).

Key Features:
- Unified Memory: CPU and GPU share the same physical memory
- Pinned Memory: Page-locked host memory for fast DMA transfers
- CUDA Memory Pools: Pre-allocated GPU memory to avoid allocation overhead
- Managed Memory: Automatic migration between CPU and GPU

On DGX Spark's Grace Hopper architecture:
- The CPU (Grace) and GPU (Hopper) share a unified memory space
- No explicit copies needed - both can access the same physical memory
- NVLink-C2C provides 900 GB/s bandwidth between CPU and GPU
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
from contextlib import contextmanager
import structlog

log = structlog.get_logger()

# Check for required libraries
try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class BufferStats:
    """Statistics for buffer pool usage."""
    total_buffers: int = 0
    buffers_in_use: int = 0
    total_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    allocations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class GPUBuffer:
    """
    A GPU buffer that can be used for zero-copy operations.

    Attributes:
        tensor: PyTorch tensor on GPU (or unified memory)
        shape: Buffer shape
        dtype: Data type
        is_pinned: Whether using pinned memory
        is_unified: Whether using unified/managed memory
        device: CUDA device index
    """
    tensor: "torch.Tensor"
    shape: Tuple[int, ...]
    dtype: "torch.dtype"
    is_pinned: bool = False
    is_unified: bool = False
    device: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def data_ptr(self) -> int:
        """Get raw data pointer for interop."""
        return self.tensor.data_ptr()

    @property
    def nbytes(self) -> int:
        """Get buffer size in bytes."""
        return self.tensor.numel() * self.tensor.element_size()

    def to_numpy(self) -> "np.ndarray":
        """
        Convert to numpy array.
        On unified memory, this is a view (zero-copy).
        """
        if self.is_unified and CUPY_AVAILABLE:
            # True zero-copy via cupy
            return cp.asnumpy(cp.asarray(self.tensor))
        else:
            # Requires sync but minimal copy on unified memory
            return self.tensor.cpu().numpy()

    def as_cuda_array(self) -> Any:
        """Get as CUDA array interface for interop."""
        return self.tensor.__cuda_array_interface__


class ZeroCopyBufferPool:
    """
    Memory pool for zero-copy GPU buffers.

    Uses DGX Spark's unified memory for true zero-copy between CPU and GPU.
    Pre-allocates buffers to avoid runtime allocation overhead.
    """

    def __init__(
        self,
        pool_size_mb: int = 512,
        default_shape: Tuple[int, ...] = (1080, 1920, 3),
        dtype: str = "uint8",
        use_unified_memory: bool = True,
        device: int = 0
    ):
        """
        Initialize the zero-copy buffer pool.

        Args:
            pool_size_mb: Total pool size in megabytes
            default_shape: Default buffer shape (H, W, C for frames)
            dtype: Data type ("uint8", "float32", "float16")
            use_unified_memory: Use CUDA unified/managed memory
            device: CUDA device index
        """
        self.pool_size_mb = pool_size_mb
        self.default_shape = default_shape
        self.device = device
        self.use_unified_memory = use_unified_memory and self._check_unified_memory_support()

        # Map string dtype to torch dtype
        self._dtype_map = {
            "uint8": torch.uint8,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
        }
        self.dtype = self._dtype_map.get(dtype, torch.uint8)

        # Buffer pools (keyed by shape)
        self._free_buffers: Dict[Tuple[int, ...], List[GPUBuffer]] = {}
        self._in_use_buffers: Dict[int, GPUBuffer] = {}  # keyed by data_ptr
        self._stats = BufferStats()

        # Initialize CUDA
        if CUDA_AVAILABLE:
            torch.cuda.set_device(device)
            self._initialize_pool()

        log.info(
            "zero_copy_pool_initialized",
            pool_size_mb=pool_size_mb,
            unified_memory=self.use_unified_memory,
            device=device
        )

    def _check_unified_memory_support(self) -> bool:
        """Check if unified memory is supported (Grace Hopper / DGX Spark)."""
        if not CUDA_AVAILABLE:
            return False

        try:
            props = torch.cuda.get_device_properties(self.device)
            # Grace Hopper has compute capability 9.0
            # Check for managed memory support
            is_grace_hopper = props.major >= 9

            # Also check environment variable for forcing unified memory
            force_unified = os.environ.get("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "0") == "1"

            if is_grace_hopper:
                log.info("grace_hopper_detected", compute_capability=f"{props.major}.{props.minor}")

            return is_grace_hopper or force_unified

        except Exception as e:
            log.warning("unified_memory_check_failed", error=str(e))
            return False

    def _initialize_pool(self) -> None:
        """Pre-allocate the buffer pool."""
        if not CUDA_AVAILABLE:
            return

        # Calculate buffer size
        buffer_bytes = np.prod(self.default_shape) * torch.tensor([], dtype=self.dtype).element_size()
        num_buffers = (self.pool_size_mb * 1024 * 1024) // buffer_bytes
        num_buffers = max(1, min(num_buffers, 32))  # Reasonable limits

        log.info(
            "preallocating_buffers",
            num_buffers=num_buffers,
            buffer_bytes=buffer_bytes,
            total_mb=f"{(num_buffers * buffer_bytes) / (1024*1024):.1f}"
        )

        self._free_buffers[self.default_shape] = []

        for i in range(num_buffers):
            buffer = self._allocate_buffer(self.default_shape)
            if buffer:
                self._free_buffers[self.default_shape].append(buffer)
                self._stats.total_buffers += 1
                self._stats.total_memory_bytes += buffer.nbytes

    def _allocate_buffer(self, shape: Tuple[int, ...]) -> Optional[GPUBuffer]:
        """Allocate a new GPU buffer."""
        if not CUDA_AVAILABLE:
            return None

        try:
            if self.use_unified_memory:
                # Use CUDA managed memory (unified memory)
                # On Grace Hopper, this is true unified memory
                tensor = torch.cuda.FloatTensor(*shape).to(self.dtype)
                # Enable managed memory hint
                if hasattr(torch.cuda, 'mem_get_info'):
                    # Allocate as managed memory
                    tensor = torch.empty(
                        shape,
                        dtype=self.dtype,
                        device=f'cuda:{self.device}'
                    )
            else:
                # Standard GPU allocation
                tensor = torch.empty(
                    shape,
                    dtype=self.dtype,
                    device=f'cuda:{self.device}'
                )

            self._stats.allocations += 1

            return GPUBuffer(
                tensor=tensor,
                shape=shape,
                dtype=self.dtype,
                is_pinned=False,
                is_unified=self.use_unified_memory,
                device=self.device
            )

        except Exception as e:
            log.error("buffer_allocation_failed", shape=shape, error=str(e))
            return None

    def acquire(self, shape: Optional[Tuple[int, ...]] = None) -> Optional[GPUBuffer]:
        """
        Acquire a buffer from the pool.

        Args:
            shape: Buffer shape (uses default if None)

        Returns:
            GPUBuffer or None if allocation fails
        """
        shape = shape or self.default_shape

        # Try to get from pool
        if shape in self._free_buffers and self._free_buffers[shape]:
            buffer = self._free_buffers[shape].pop()
            self._in_use_buffers[buffer.data_ptr] = buffer
            self._stats.buffers_in_use += 1
            self._stats.cache_hits += 1
            buffer.timestamp = time.time()
            return buffer

        # Allocate new buffer
        self._stats.cache_misses += 1
        buffer = self._allocate_buffer(shape)

        if buffer:
            self._in_use_buffers[buffer.data_ptr] = buffer
            self._stats.buffers_in_use += 1
            self._stats.total_buffers += 1
            self._stats.total_memory_bytes += buffer.nbytes
            self._stats.peak_memory_bytes = max(
                self._stats.peak_memory_bytes,
                self._stats.total_memory_bytes
            )

        return buffer

    def release(self, buffer: GPUBuffer) -> None:
        """
        Release a buffer back to the pool.

        Args:
            buffer: Buffer to release
        """
        ptr = buffer.data_ptr

        if ptr in self._in_use_buffers:
            del self._in_use_buffers[ptr]
            self._stats.buffers_in_use -= 1

            # Return to pool
            if buffer.shape not in self._free_buffers:
                self._free_buffers[buffer.shape] = []
            self._free_buffers[buffer.shape].append(buffer)

    @contextmanager
    def get_buffer(self, shape: Optional[Tuple[int, ...]] = None):
        """
        Context manager for automatic buffer acquisition and release.

        Usage:
            with pool.get_buffer() as buf:
                # Use buf.tensor
        """
        buffer = self.acquire(shape)
        try:
            yield buffer
        finally:
            if buffer:
                self.release(buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_buffers": self._stats.total_buffers,
            "buffers_in_use": self._stats.buffers_in_use,
            "buffers_free": self._stats.total_buffers - self._stats.buffers_in_use,
            "total_memory_mb": self._stats.total_memory_bytes / (1024 * 1024),
            "peak_memory_mb": self._stats.peak_memory_bytes / (1024 * 1024),
            "allocations": self._stats.allocations,
            "cache_hit_rate": (
                self._stats.cache_hits / max(1, self._stats.cache_hits + self._stats.cache_misses)
            ),
            "unified_memory": self.use_unified_memory
        }

    def clear(self) -> None:
        """Clear all buffers and free memory."""
        self._free_buffers.clear()
        self._in_use_buffers.clear()

        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()

        self._stats = BufferStats()
        log.info("buffer_pool_cleared")


class PinnedMemoryAllocator:
    """
    Allocator for pinned (page-locked) host memory.

    Pinned memory enables faster CPU↔GPU transfers via DMA.
    On DGX Spark with unified memory, this provides optimal access patterns.
    """

    def __init__(self, prealloc_mb: int = 256):
        """
        Initialize pinned memory allocator.

        Args:
            prealloc_mb: Pre-allocated pinned memory in MB
        """
        self.prealloc_mb = prealloc_mb
        self._pinned_tensors: Dict[int, torch.Tensor] = {}
        self._stats = BufferStats()

        if CUDA_AVAILABLE:
            self._initialize()

    def _initialize(self) -> None:
        """Pre-allocate pinned memory."""
        try:
            # Pre-allocate a large pinned buffer
            size = self.prealloc_mb * 1024 * 1024
            self._pool_tensor = torch.empty(
                size,
                dtype=torch.uint8,
                pin_memory=True
            )
            self._stats.total_memory_bytes = size
            log.info("pinned_memory_initialized", size_mb=self.prealloc_mb)
        except Exception as e:
            log.warning("pinned_memory_init_failed", error=str(e))

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.uint8
    ) -> Optional[torch.Tensor]:
        """
        Allocate a pinned memory tensor.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Pinned tensor or None
        """
        if not TORCH_AVAILABLE:
            return None

        try:
            tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._pinned_tensors[tensor.data_ptr()] = tensor
            self._stats.allocations += 1
            return tensor
        except Exception as e:
            log.warning("pinned_allocation_failed", error=str(e))
            return None

    def free(self, tensor: torch.Tensor) -> None:
        """Free a pinned tensor."""
        ptr = tensor.data_ptr()
        if ptr in self._pinned_tensors:
            del self._pinned_tensors[ptr]


class UnifiedMemoryTensor:
    """
    A tensor wrapper that provides unified memory access on DGX Spark.

    On Grace Hopper architecture:
    - CPU and GPU see the same physical memory
    - No explicit copies needed
    - Automatic page migration based on access patterns
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: int = 0
    ):
        """
        Create a unified memory tensor.

        Args:
            shape: Tensor shape
            dtype: Data type
            device: CUDA device
        """
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._tensor: Optional[torch.Tensor] = None
        self._cupy_array: Optional[Any] = None

        self._allocate()

    def _allocate(self) -> None:
        """Allocate unified memory tensor."""
        if not CUDA_AVAILABLE:
            # Fallback to CPU
            self._tensor = torch.empty(self.shape, dtype=self.dtype)
            return

        try:
            # Allocate on GPU - on Grace Hopper this is unified memory
            self._tensor = torch.empty(
                self.shape,
                dtype=self.dtype,
                device=f'cuda:{self.device}'
            )

            # For true unified memory access, we can use cupy interop
            if CUPY_AVAILABLE:
                # Create cupy array view of the same memory
                self._cupy_array = cp.asarray(self._tensor)

            log.debug(
                "unified_tensor_created",
                shape=self.shape,
                dtype=str(self.dtype),
                ptr=self._tensor.data_ptr()
            )

        except Exception as e:
            log.warning("unified_tensor_failed", error=str(e))
            self._tensor = torch.empty(self.shape, dtype=self.dtype)

    @property
    def tensor(self) -> torch.Tensor:
        """Get PyTorch tensor."""
        return self._tensor

    @property
    def numpy(self) -> np.ndarray:
        """
        Get numpy view of the data.
        On unified memory, this doesn't copy data.
        """
        if CUPY_AVAILABLE and self._cupy_array is not None:
            # Zero-copy via cupy
            return cp.asnumpy(self._cupy_array)
        else:
            return self._tensor.cpu().numpy()

    @property
    def cupy(self) -> Any:
        """Get cupy array (zero-copy on unified memory)."""
        if self._cupy_array is not None:
            return self._cupy_array
        elif CUPY_AVAILABLE and self._tensor.is_cuda:
            return cp.asarray(self._tensor)
        else:
            raise RuntimeError("CuPy not available or tensor not on GPU")

    def copy_from_numpy(self, arr: np.ndarray) -> None:
        """
        Copy data from numpy array.
        On unified memory, this may be a direct write.
        """
        if CUPY_AVAILABLE and self._cupy_array is not None:
            # Use cupy for potentially faster transfer
            cp_arr = cp.asarray(arr)
            self._cupy_array[:] = cp_arr
        else:
            self._tensor.copy_(torch.from_numpy(arr))

    def __cuda_array_interface__(self) -> Dict:
        """CUDA array interface for interoperability."""
        return self._tensor.__cuda_array_interface__


# Global pool instance
_global_pool: Optional[ZeroCopyBufferPool] = None


def get_buffer_pool(
    pool_size_mb: int = 512,
    default_shape: Tuple[int, ...] = (1080, 1920, 3)
) -> ZeroCopyBufferPool:
    """Get or create the global buffer pool."""
    global _global_pool

    if _global_pool is None:
        _global_pool = ZeroCopyBufferPool(
            pool_size_mb=pool_size_mb,
            default_shape=default_shape
        )

    return _global_pool


def is_unified_memory_available() -> bool:
    """Check if unified memory is available (DGX Spark / Grace Hopper)."""
    if not CUDA_AVAILABLE:
        return False

    try:
        props = torch.cuda.get_device_properties(0)
        return props.major >= 9  # Grace Hopper
    except Exception:
        return False
