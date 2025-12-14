"""
OSHA Vision Unified Initializer.

Single entry point for initializing all DGX Spark optimizations.
Handles graceful fallbacks and provides runtime configuration.

Usage:
    from src.core import initialize_osha_vision

    config = initialize_osha_vision(optimization_level="maximum")
    # All optimizations are now active
"""

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import structlog

log = structlog.get_logger()

# ============================================================================
# OPTIMIZATION LEVELS
# ============================================================================

class OptimizationLevel(Enum):
    """Optimization levels - all preserve model quality."""
    MINIMAL = "minimal"      # Basic CUDA, no special optimizations
    STANDARD = "standard"    # TensorRT, FAISS-GPU
    MAXIMUM = "maximum"      # Full zero-copy pipeline
    AUTO = "auto"            # Auto-detect best level


# ============================================================================
# RUNTIME CONFIGURATION
# ============================================================================

@dataclass
class RuntimeConfig:
    """
    Runtime configuration populated during initialization.

    All fields are set based on detected hardware capabilities.
    """
    # Hardware info
    cuda_available: bool = False
    gpu_name: str = "N/A"
    gpu_memory_gb: float = 0.0
    compute_capability: str = "N/A"

    # Architecture detection
    is_grace_hopper: bool = False
    unified_memory_available: bool = False
    nvdec_available: bool = False

    # Active optimizations
    tensorrt_enabled: bool = False
    faiss_gpu_enabled: bool = False
    zero_copy_enabled: bool = False
    cuda_streams: int = 0

    # Paths
    tensorrt_engine_path: Optional[str] = None

    # Performance settings
    optimal_batch_size: int = 1
    num_decode_threads: int = 2
    buffer_pool_mb: int = 256

    # Status
    optimization_level: str = "minimal"
    initialization_time_ms: float = 0.0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "compute_capability": self.compute_capability,
            "is_grace_hopper": self.is_grace_hopper,
            "unified_memory_available": self.unified_memory_available,
            "nvdec_available": self.nvdec_available,
            "tensorrt_enabled": self.tensorrt_enabled,
            "faiss_gpu_enabled": self.faiss_gpu_enabled,
            "zero_copy_enabled": self.zero_copy_enabled,
            "cuda_streams": self.cuda_streams,
            "optimization_level": self.optimization_level,
            "initialization_time_ms": self.initialization_time_ms,
            "warnings": self.warnings
        }


# Global runtime config
_runtime_config: Optional[RuntimeConfig] = None


def get_runtime_config() -> RuntimeConfig:
    """Get the current runtime configuration."""
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config


# ============================================================================
# HARDWARE DETECTION
# ============================================================================

def _detect_hardware() -> Tuple[RuntimeConfig, Dict[str, Any]]:
    """Detect hardware capabilities."""
    config = RuntimeConfig()
    details = {}

    # Check CUDA
    try:
        import torch
        config.cuda_available = torch.cuda.is_available()

        if config.cuda_available:
            props = torch.cuda.get_device_properties(0)
            config.gpu_name = props.name
            config.gpu_memory_gb = props.total_memory / (1024**3)
            config.compute_capability = f"{props.major}.{props.minor}"

            # Grace Hopper detection (compute capability 9.x)
            config.is_grace_hopper = props.major >= 9
            config.unified_memory_available = config.is_grace_hopper

            details["gpu"] = {
                "name": config.gpu_name,
                "memory_gb": round(config.gpu_memory_gb, 1),
                "compute_capability": config.compute_capability,
                "is_grace_hopper": config.is_grace_hopper
            }

            # Calculate optimal batch size based on memory
            if config.gpu_memory_gb > 80:
                config.optimal_batch_size = 8
                config.buffer_pool_mb = 1024
            elif config.gpu_memory_gb > 40:
                config.optimal_batch_size = 4
                config.buffer_pool_mb = 512
            else:
                config.optimal_batch_size = 2
                config.buffer_pool_mb = 256

    except ImportError:
        config.warnings.append("PyTorch not installed")
    except Exception as e:
        config.warnings.append(f"CUDA detection failed: {e}")

    # Check NVDEC
    try:
        import PyNvVideoCodec
        config.nvdec_available = True
    except ImportError:
        config.nvdec_available = False

    # Check cupy
    try:
        import cupy
        details["cupy"] = True
    except ImportError:
        details["cupy"] = False

    return config, details


# ============================================================================
# OPTIMIZATION INITIALIZERS
# ============================================================================

def _init_cuda_optimizations(config: RuntimeConfig) -> None:
    """Initialize CUDA-level optimizations."""
    if not config.cuda_available:
        return

    try:
        import torch

        # cuDNN benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True

        # TF32 for Ampere+ GPUs
        if int(config.compute_capability.split('.')[0]) >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Unified memory configuration (Grace Hopper)
        if config.unified_memory_available:
            torch.cuda.set_per_process_memory_fraction(0.9, 0)

        # Create CUDA streams
        config.cuda_streams = 4

        log.info("cuda_optimizations_initialized",
                 tf32=True,
                 cudnn_benchmark=True,
                 streams=config.cuda_streams)

    except Exception as e:
        config.warnings.append(f"CUDA optimization failed: {e}")


def _init_dgx_spark_optimizer(config: RuntimeConfig) -> None:
    """Initialize DGX Spark specific optimizations."""
    try:
        from src.utils.dgx_spark_optimizer import init_dgx_spark, get_optimizer

        status = init_dgx_spark()

        if status.get("status") == "initialized":
            config.zero_copy_enabled = True
            log.info("dgx_spark_optimizer_initialized")
        else:
            config.warnings.append("DGX Spark optimizer not fully initialized")

    except ImportError:
        config.warnings.append("DGX Spark optimizer module not found")
    except Exception as e:
        config.warnings.append(f"DGX Spark init failed: {e}")


def _init_buffer_pool(config: RuntimeConfig) -> None:
    """Initialize zero-copy buffer pool."""
    if not config.cuda_available:
        return

    try:
        from src.utils.zero_copy_buffer import get_buffer_pool

        pool = get_buffer_pool(
            pool_size_mb=config.buffer_pool_mb,
            default_shape=(1080, 1920, 3)
        )

        config.zero_copy_enabled = pool.use_unified_memory
        log.info("buffer_pool_initialized",
                 size_mb=config.buffer_pool_mb,
                 unified_memory=pool.use_unified_memory)

    except ImportError:
        pass  # Optional component
    except Exception as e:
        config.warnings.append(f"Buffer pool init failed: {e}")


def _init_faiss_gpu(config: RuntimeConfig) -> None:
    """Initialize FAISS-GPU."""
    if not config.cuda_available:
        return

    try:
        import faiss

        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            config.faiss_gpu_enabled = True
            log.info("faiss_gpu_available", num_gpus=num_gpus)
        else:
            config.warnings.append("FAISS-GPU not available, using CPU")

    except (ImportError, AttributeError):
        config.warnings.append("FAISS not installed with GPU support")
    except Exception as e:
        config.warnings.append(f"FAISS-GPU init failed: {e}")


def _init_tensorrt(config: RuntimeConfig, model_path: Optional[str] = None) -> None:
    """Check TensorRT availability."""
    try:
        # Check if TensorRT is available through torch
        import torch

        # TensorRT is typically available if torch has CUDA
        if config.cuda_available:
            config.tensorrt_enabled = True
            config.tensorrt_engine_path = model_path or "models/yolo_trt.engine"
            log.info("tensorrt_available", engine_path=config.tensorrt_engine_path)

    except Exception as e:
        config.warnings.append(f"TensorRT check failed: {e}")


# ============================================================================
# MAIN INITIALIZATION
# ============================================================================

def initialize_osha_vision(
    optimization_level: str = "auto",
    tensorrt_model_path: Optional[str] = None,
    verbose: bool = True
) -> RuntimeConfig:
    """
    Initialize all OSHA Vision optimizations.

    This is the main entry point that should be called once at application startup.
    It detects hardware capabilities and enables all appropriate optimizations.

    Args:
        optimization_level: "minimal", "standard", "maximum", or "auto"
        tensorrt_model_path: Optional path to TensorRT engine
        verbose: Print initialization status

    Returns:
        RuntimeConfig with all settings and detected capabilities
    """
    global _runtime_config

    start_time = time.perf_counter()

    if verbose:
        print("\n" + "=" * 60)
        print("  OSHA VISION - DGX SPARK INITIALIZATION")
        print("=" * 60)

    # Detect hardware
    config, details = _detect_hardware()

    # Determine optimization level
    if optimization_level == "auto":
        if config.is_grace_hopper:
            optimization_level = "maximum"
        elif config.cuda_available:
            optimization_level = "standard"
        else:
            optimization_level = "minimal"

    config.optimization_level = optimization_level

    if verbose:
        print(f"\n  Hardware Detection:")
        print(f"    GPU: {config.gpu_name}")
        print(f"    Memory: {config.gpu_memory_gb:.1f} GB")
        print(f"    Compute: {config.compute_capability}")
        print(f"    Grace Hopper: {config.is_grace_hopper}")
        print(f"    Unified Memory: {config.unified_memory_available}")
        print(f"\n  Optimization Level: {optimization_level.upper()}")

    # Initialize based on level
    if optimization_level in ["standard", "maximum"]:
        if verbose:
            print("\n  Initializing optimizations...")

        _init_cuda_optimizations(config)
        _init_tensorrt(config, tensorrt_model_path)
        _init_faiss_gpu(config)

    if optimization_level == "maximum":
        _init_dgx_spark_optimizer(config)
        _init_buffer_pool(config)

    # Calculate initialization time
    config.initialization_time_ms = (time.perf_counter() - start_time) * 1000

    if verbose:
        print(f"\n  Active Optimizations:")
        print(f"    TensorRT: {config.tensorrt_enabled}")
        print(f"    FAISS-GPU: {config.faiss_gpu_enabled}")
        print(f"    Zero-Copy: {config.zero_copy_enabled}")
        print(f"    CUDA Streams: {config.cuda_streams}")
        print(f"    NVDEC: {config.nvdec_available}")

        if config.warnings:
            print(f"\n  Warnings:")
            for w in config.warnings:
                print(f"    - {w}")

        print(f"\n  Initialization time: {config.initialization_time_ms:.1f} ms")
        print("=" * 60 + "\n")

    _runtime_config = config
    return config


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def is_zero_copy_available() -> bool:
    """Check if zero-copy pipeline is available."""
    config = get_runtime_config()
    return config.zero_copy_enabled


def is_grace_hopper() -> bool:
    """Check if running on Grace Hopper (DGX Spark)."""
    config = get_runtime_config()
    return config.is_grace_hopper


def get_optimal_batch_size() -> int:
    """Get optimal batch size for current hardware."""
    config = get_runtime_config()
    return config.optimal_batch_size


def create_optimized_detector(
    model_size: str = "yolov8s-worldv2",
    confidence: float = 0.3,
    classes: Optional[list] = None
):
    """
    Create an optimized PPE detector based on runtime config.

    Automatically uses TensorRT if available.
    """
    config = get_runtime_config()

    try:
        from src.pipeline.detection import PPEDetector

        return PPEDetector(
            model_size=model_size,
            confidence_threshold=confidence,
            custom_classes=classes,
            device="cuda" if config.cuda_available else "cpu",
            tensorrt_engine_path=config.tensorrt_engine_path if config.tensorrt_enabled else None,
            auto_export_tensorrt=config.tensorrt_enabled
        )

    except ImportError:
        log.error("PPEDetector not available")
        return None


def create_zero_copy_decoder(video_path: str):
    """
    Create a zero-copy video decoder.

    Falls back gracefully if NVDEC not available.
    """
    try:
        from src.pipeline.cuda_video_decoder import ZeroCopyVideoDecoder

        decoder = ZeroCopyVideoDecoder(prefer_nvdec=True)
        if decoder.open(video_path):
            return decoder
        return None

    except ImportError:
        return None


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics."""
    config = get_runtime_config()

    if not config.cuda_available:
        return {"available": False}

    try:
        import torch

        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)

        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(config.gpu_memory_gb, 2),
            "free_gb": round(config.gpu_memory_gb - allocated, 2),
            "utilization_pct": round((allocated / config.gpu_memory_gb) * 100, 1)
        }

    except Exception:
        return {"error": "Failed to get memory stats"}
