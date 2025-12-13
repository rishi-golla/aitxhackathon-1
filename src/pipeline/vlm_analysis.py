"""
VLM Safety Analysis Module.

Uses Qwen2.5-VL-7B for scene understanding and safety analysis.
"""

import base64
import json
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional
import numpy as np
import structlog

log = structlog.get_logger()

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig
    )
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    log.warning("transformers_not_installed")


@dataclass
class SceneDescription:
    """Description of analyzed scene."""
    workers_detected: int
    activities: list[str]  # ["welding", "lifting", "walking"]
    hazards_observed: list[str]  # ["sparks", "exposed wiring"]
    ppe_status: dict[str, list[str]]  # {"worker_1": ["hardhat", "vest"]}
    zone_assessment: str  # "Welding bay with active work"
    safety_concerns: list[str]  # Potential issues
    confidence: float
    inference_time_ms: float
    raw_response: Optional[str] = None


# VLM analysis prompt template
SAFETY_ANALYSIS_PROMPT = """You are a factory safety analyst reviewing security camera footage. Analyze this frame and provide a safety assessment.

DETECTED OBJECTS FROM YOLO:
{detection_list}

ZONE CONTEXT:
{zone_info}

Analyze the scene and respond in JSON format:
{{
    "workers_detected": <number>,
    "activities": ["<activity1>", "<activity2>"],
    "ppe_status": {{
        "worker_1": ["<ppe_item1>", "<ppe_item2>"],
        "worker_2": ["<ppe_item1>"]
    }},
    "hazards_observed": ["<hazard1>", "<hazard2>"],
    "safety_concerns": ["<concern1>", "<concern2>"],
    "zone_assessment": "<brief assessment>"
}}

Be specific and factual. Only list what you can actually see in the image."""


class VLMAnalyzer:
    """
    Vision Language Model analyzer for factory safety scenes.

    Uses Qwen2.5-VL-7B with 4-bit quantization for efficient inference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization: str = "4bit",
        device: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.3
    ):
        """
        Initialize VLM analyzer.

        Args:
            model_name: HuggingFace model identifier
            quantization: Quantization mode ("none", "4bit", "8bit")
            device: Device to run on ("auto", "cuda", "cpu")
            max_tokens: Maximum tokens for generation
            temperature: Generation temperature
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required. Install with: pip install transformers")

        if not TORCH_AVAILABLE:
            raise ImportError("torch package required. Install with: pip install torch")

        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._model = None
        self._processor = None
        self._loaded = False

        log.info(
            "vlm_analyzer_init",
            model=model_name,
            quantization=quantization,
            device=device
        )

    def load_model(self) -> None:
        """Load the VLM model with specified quantization."""
        if self._loaded:
            return

        log.info("vlm_loading_model", model=self.model_name)
        start_time = time.time()

        # Configure quantization
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = None

        # Load model
        model_kwargs = {
            "device_map": self.device,
            "trust_remote_code": True
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        # Try to use flash attention if available
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        load_time = time.time() - start_time
        self._loaded = True

        log.info(
            "vlm_model_loaded",
            model=self.model_name,
            load_time_s=round(load_time, 2)
        )

    def _prepare_image(self, frame: np.ndarray) -> Image.Image:
        """Convert numpy frame to PIL Image."""
        from PIL import Image
        import cv2

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string."""
        import cv2

        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def generate_prompt(
        self,
        detections: list,  # List of Detection objects
        zone_context: str
    ) -> str:
        """
        Create structured prompt for safety analysis.

        Args:
            detections: List of Detection objects from YOLO
            zone_context: Description of the zone

        Returns:
            Formatted prompt string
        """
        # Format detections
        detection_lines = []
        for det in detections:
            line = f"- {det.class_name} (confidence: {det.confidence:.2f})"
            if det.track_id:
                line += f" [ID: {det.track_id}]"
            detection_lines.append(line)

        detection_list = "\n".join(detection_lines) if detection_lines else "No objects detected"

        return SAFETY_ANALYSIS_PROMPT.format(
            detection_list=detection_list,
            zone_info=zone_context or "General work area"
        )

    def _parse_response(self, response: str) -> dict:
        """Parse JSON response from model."""
        # Try to extract JSON from response
        try:
            # First try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return default structure if parsing fails
        log.warning("vlm_response_parse_failed", response=response[:200])
        return {
            "workers_detected": 0,
            "activities": [],
            "ppe_status": {},
            "hazards_observed": [],
            "safety_concerns": ["Unable to parse VLM response"],
            "zone_assessment": "Analysis incomplete"
        }

    def analyze_frame(
        self,
        frame: np.ndarray,
        detections: list,
        zone_context: str = ""
    ) -> SceneDescription:
        """
        Analyze a frame for safety compliance.

        Args:
            frame: BGR image as numpy array
            detections: List of Detection objects
            zone_context: Description of the zone

        Returns:
            SceneDescription with analysis results
        """
        if not self._loaded:
            self.load_model()

        start_time = time.time()

        # Prepare image
        image = self._prepare_image(frame)

        # Generate prompt
        text_prompt = self.generate_prompt(detections, zone_context)

        # Prepare inputs for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]

        # Process inputs
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )

        # Move to device
        if self.device == "auto" or self.device == "cuda":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )

        # Decode response
        response = self._processor.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )

        inference_time = (time.time() - start_time) * 1000

        # Parse response
        parsed = self._parse_response(response)

        return SceneDescription(
            workers_detected=parsed.get("workers_detected", 0),
            activities=parsed.get("activities", []),
            hazards_observed=parsed.get("hazards_observed", []),
            ppe_status=parsed.get("ppe_status", {}),
            zone_assessment=parsed.get("zone_assessment", ""),
            safety_concerns=parsed.get("safety_concerns", []),
            confidence=0.8,  # Placeholder - could be derived from response
            inference_time_ms=inference_time,
            raw_response=response
        )

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model:
            del self._model
            self._model = None

        if self._processor:
            del self._processor
            self._processor = None

        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()

        self._loaded = False
        log.info("vlm_model_unloaded")

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "device": self.device,
            "loaded": self._loaded,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class MockVLMAnalyzer:
    """
    Mock VLM analyzer for testing without GPU.

    Returns plausible mock responses based on detection input.
    """

    def __init__(self, **kwargs):
        self._loaded = True

    def load_model(self) -> None:
        pass

    def analyze_frame(
        self,
        frame: np.ndarray,
        detections: list,
        zone_context: str = ""
    ) -> SceneDescription:
        """Generate mock analysis based on detections."""
        # Count workers
        workers = [d for d in detections if d.class_name.lower() in {"person", "worker", "human"}]

        # Extract PPE items
        ppe_items = {}
        ppe_classes = {"hardhat", "safety vest", "safety glasses", "gloves", "face shield"}

        for i, worker in enumerate(workers):
            worker_id = f"worker_{i+1}"
            ppe_items[worker_id] = []

            # Check for PPE near this worker
            for det in detections:
                if det.class_name.lower() in ppe_classes:
                    ppe_items[worker_id].append(det.class_name)

        # Generate concerns based on zone
        concerns = []
        if "welding" in zone_context.lower():
            # Check if any worker has welding PPE
            has_welding_ppe = False
            for worker_id, items in ppe_items.items():
                for item in items:
                    if "welding" in item.lower() or "mask" in item.lower():
                        has_welding_ppe = True
                        break
            if not has_welding_ppe:
                concerns.append("Welding PPE may be insufficient")

        return SceneDescription(
            workers_detected=len(workers),
            activities=["working", "standing"] if workers else [],
            hazards_observed=[],
            ppe_status=ppe_items,
            zone_assessment=zone_context or "General work area observed",
            safety_concerns=concerns,
            confidence=0.9,
            inference_time_ms=50.0,
            raw_response="[MOCK RESPONSE]"
        )

    def unload_model(self) -> None:
        pass

    def get_model_info(self) -> dict:
        return {"model_name": "mock", "loaded": True}
