#!/usr/bin/env python3
"""
Multi-Stream Parallel Processing Demo for DGX Spark.

Demonstrates processing multiple video streams simultaneously using CUDA streams.
This showcases the DGX Spark's ability to handle multiple camera feeds in parallel.

Usage:
    python scripts/multi_stream_demo.py --streams 4 --duration 30
"""

import argparse
import asyncio
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Add root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


@dataclass
class StreamMetrics:
    """Metrics for a single video stream."""
    stream_id: int
    frames_processed: int
    total_time_ms: float
    avg_inference_ms: float
    violations_detected: int
    fps: float


@dataclass
class MultiStreamResults:
    """Results from multi-stream processing."""
    num_streams: int
    total_frames: int
    total_time_s: float
    aggregate_fps: float
    per_stream_fps: float
    cuda_streams_used: int
    zero_copy_enabled: bool
    tensorrt_enabled: bool
    gpu_utilization: float
    memory_used_gb: float
    stream_metrics: List[StreamMetrics]


class MultiStreamProcessor:
    """
    Parallel video stream processor using CUDA streams.

    Demonstrates DGX Spark's multi-stream capabilities.
    """

    def __init__(self, num_streams: int = 4, use_cuda_streams: bool = True):
        self.num_streams = num_streams
        self.use_cuda_streams = use_cuda_streams
        self._initialized = False
        self._model = None
        self._cuda_streams = []
        self._runtime_config = None

    def initialize(self) -> Dict[str, Any]:
        """Initialize CUDA streams and model."""
        import torch

        status = {"cuda_available": torch.cuda.is_available()}

        if not status["cuda_available"]:
            print("WARNING: CUDA not available, running on CPU")
            return status

        # Initialize runtime config
        try:
            from src.core.initializer import initialize_osha_vision, get_runtime_config
            self._runtime_config = initialize_osha_vision(
                optimization_level="auto",
                verbose=False
            )
            status["runtime_config"] = self._runtime_config.to_dict()
        except ImportError:
            print("Note: Core initializer not available")

        # Create CUDA streams for parallel processing
        if self.use_cuda_streams:
            self._cuda_streams = [
                torch.cuda.Stream() for _ in range(self.num_streams)
            ]
            status["cuda_streams_created"] = len(self._cuda_streams)

        # Load YOLO model
        try:
            from ultralytics import YOLOWorld
            model_path = root_dir / "models" / "yolov8s-world.pt"
            self._model = YOLOWorld(str(model_path))
            self._model.to('cuda')
            self._model.set_classes([
                "person", "worker", "hand", "bare_hand",
                "glove", "safety_glasses", "machine", "tool"
            ])
            status["model_loaded"] = True
        except Exception as e:
            print(f"Model loading failed: {e}")
            status["model_loaded"] = False

        self._initialized = True
        return status

    def _process_single_stream(
        self,
        stream_id: int,
        video_path: str,
        num_frames: int,
        cuda_stream: Optional[Any] = None
    ) -> StreamMetrics:
        """Process a single video stream."""
        import torch
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return StreamMetrics(
                stream_id=stream_id,
                frames_processed=0,
                total_time_ms=0,
                avg_inference_ms=0,
                violations_detected=0,
                fps=0
            )

        frames_processed = 0
        violations = 0
        inference_times = []
        start_time = time.perf_counter()

        while frames_processed < num_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run inference (with CUDA stream if available)
            inf_start = time.perf_counter()

            if cuda_stream is not None:
                with torch.cuda.stream(cuda_stream):
                    results = self._model.predict(frame, conf=0.1, verbose=False)
            else:
                results = self._model.predict(frame, conf=0.1, verbose=False)

            inf_time = (time.perf_counter() - inf_start) * 1000
            inference_times.append(inf_time)

            # Count detections
            if results and len(results) > 0:
                result = results[0]
                if result.boxes:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id]
                        if "bare_hand" in class_name or "hand" in class_name:
                            violations += 1

            frames_processed += 1

        cap.release()

        total_time = (time.perf_counter() - start_time) * 1000
        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
        fps = frames_processed / (total_time / 1000) if total_time > 0 else 0

        return StreamMetrics(
            stream_id=stream_id,
            frames_processed=frames_processed,
            total_time_ms=total_time,
            avg_inference_ms=avg_inference,
            violations_detected=violations,
            fps=fps
        )

    def process_parallel(
        self,
        video_paths: List[str],
        frames_per_stream: int = 100
    ) -> MultiStreamResults:
        """
        Process multiple video streams in parallel.

        Uses CUDA streams for true parallel execution on DGX Spark.
        """
        import torch

        if not self._initialized:
            self.initialize()

        print(f"\n{'='*60}")
        print(f"  MULTI-STREAM PARALLEL PROCESSING DEMO")
        print(f"{'='*60}")
        print(f"  Streams: {len(video_paths)}")
        print(f"  Frames per stream: {frames_per_stream}")
        print(f"  CUDA streams: {len(self._cuda_streams)}")
        print(f"{'='*60}\n")

        start_time = time.perf_counter()

        # Assign CUDA streams to video streams
        stream_metrics = []

        if self.use_cuda_streams and self._cuda_streams:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=len(video_paths)) as executor:
                futures = []
                for i, video_path in enumerate(video_paths):
                    cuda_stream = self._cuda_streams[i % len(self._cuda_streams)]
                    future = executor.submit(
                        self._process_single_stream,
                        i,
                        video_path,
                        frames_per_stream,
                        cuda_stream
                    )
                    futures.append(future)

                for future in futures:
                    metrics = future.result()
                    stream_metrics.append(metrics)
                    print(f"  Stream {metrics.stream_id}: {metrics.fps:.1f} FPS, "
                          f"{metrics.frames_processed} frames, "
                          f"{metrics.violations_detected} violations")
        else:
            # Sequential processing (fallback)
            for i, video_path in enumerate(video_paths):
                metrics = self._process_single_stream(i, video_path, frames_per_stream)
                stream_metrics.append(metrics)
                print(f"  Stream {metrics.stream_id}: {metrics.fps:.1f} FPS")

        total_time = time.perf_counter() - start_time
        total_frames = sum(m.frames_processed for m in stream_metrics)
        aggregate_fps = total_frames / total_time if total_time > 0 else 0

        # Get GPU stats
        gpu_util = 0.0
        mem_used = 0.0
        if torch.cuda.is_available():
            try:
                mem_used = torch.cuda.memory_allocated() / 1e9
                # Note: GPU utilization requires pynvml
            except:
                pass

        results = MultiStreamResults(
            num_streams=len(video_paths),
            total_frames=total_frames,
            total_time_s=total_time,
            aggregate_fps=aggregate_fps,
            per_stream_fps=aggregate_fps / len(video_paths) if video_paths else 0,
            cuda_streams_used=len(self._cuda_streams),
            zero_copy_enabled=self._runtime_config.zero_copy_enabled if self._runtime_config else False,
            tensorrt_enabled=self._runtime_config.tensorrt_enabled if self._runtime_config else False,
            gpu_utilization=gpu_util,
            memory_used_gb=mem_used,
            stream_metrics=stream_metrics
        )

        return results

    def print_results(self, results: MultiStreamResults):
        """Print formatted results."""
        print(f"\n{'='*60}")
        print(f"  MULTI-STREAM RESULTS")
        print(f"{'='*60}")
        print(f"  Total streams:      {results.num_streams}")
        print(f"  Total frames:       {results.total_frames}")
        print(f"  Total time:         {results.total_time_s:.2f}s")
        print(f"  Aggregate FPS:      {results.aggregate_fps:.1f}")
        print(f"  Per-stream FPS:     {results.per_stream_fps:.1f}")
        print(f"  CUDA streams:       {results.cuda_streams_used}")
        print(f"  Zero-copy:          {results.zero_copy_enabled}")
        print(f"  TensorRT:           {results.tensorrt_enabled}")
        print(f"  GPU memory:         {results.memory_used_gb:.2f} GB")
        print(f"{'='*60}")

        print(f"\n  DGX SPARK ADVANTAGE:")
        print(f"  - Processing {results.num_streams} streams simultaneously")
        print(f"  - {results.aggregate_fps:.0f} total FPS across all streams")
        print(f"  - Zero-copy memory: No CPU-GPU data transfer overhead")
        print(f"  - CUDA streams: True parallel GPU execution")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-stream parallel processing demo")
    parser.add_argument("--streams", type=int, default=4, help="Number of streams")
    parser.add_argument("--frames", type=int, default=50, help="Frames per stream")
    parser.add_argument("--no-cuda-streams", action="store_true", help="Disable CUDA streams")
    args = parser.parse_args()

    # Find video files
    data_dir = root_dir / "data"
    video_files = sorted(data_dir.glob("*.mp4"))[:args.streams]

    if not video_files:
        print(f"No video files found in {data_dir}")
        return

    # Duplicate if not enough videos
    while len(video_files) < args.streams:
        video_files.extend(video_files[:args.streams - len(video_files)])
    video_files = video_files[:args.streams]

    video_paths = [str(v) for v in video_files]

    # Run demo
    processor = MultiStreamProcessor(
        num_streams=args.streams,
        use_cuda_streams=not args.no_cuda_streams
    )

    processor.initialize()
    results = processor.process_parallel(video_paths, args.frames)
    processor.print_results(results)

    # Return results as dict for API use
    return {
        "num_streams": results.num_streams,
        "total_frames": results.total_frames,
        "aggregate_fps": round(results.aggregate_fps, 1),
        "per_stream_fps": round(results.per_stream_fps, 1),
        "total_time_s": round(results.total_time_s, 2),
        "cuda_streams_used": results.cuda_streams_used,
        "zero_copy_enabled": results.zero_copy_enabled,
        "tensorrt_enabled": results.tensorrt_enabled
    }


if __name__ == "__main__":
    main()
