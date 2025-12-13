"""
Policy Agent for safety compliance evaluation.

Evaluates scenes against safety rules and OSHA standards.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
import structlog

from src.pipeline.vlm_analysis import SceneDescription
from src.zones.zone_manager import SafetyZone, ZoneType
from src.rag.osha_rag import OshaRAG, get_osha_context

log = structlog.get_logger()


@dataclass
class PolicyDecision:
    """Decision from policy evaluation."""
    is_violation: bool
    violation_type: Optional[str]  # "missing_ppe", "restricted_zone", "overcrowded"
    severity: str  # "info", "warning", "critical"
    missing_items: list[str] = field(default_factory=list)
    osha_references: list[str] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0
    evaluation_time_ms: float = 0.0
    recommendations: list[str] = field(default_factory=list)


# Severity mappings
VIOLATION_SEVERITY = {
    "missing_ppe": {
        ZoneType.HAZARD: "critical",
        ZoneType.PPE_REQUIRED: "warning",
        ZoneType.MACHINE: "warning",
        ZoneType.RESTRICTED: "critical",
        ZoneType.SAFE: "info"
    },
    "restricted_area": "critical",
    "overcrowded": "warning"
}


class PolicyAgent:
    """
    Policy agent for evaluating safety compliance.

    Checks scene observations against zone rules and OSHA standards.
    Provides citations and explanations for violations.
    """

    def __init__(
        self,
        osha_rag: Optional[OshaRAG] = None,
        strict_mode: bool = True
    ):
        """
        Initialize policy agent.

        Args:
            osha_rag: OSHA RAG system for regulation lookup
            strict_mode: If True, flag all potential violations
        """
        self._rag = osha_rag
        self.strict_mode = strict_mode

        self._system_prompt = """You are a safety policy evaluator.
Your job is to check observations against zone rules and OSHA standards.
Be strict but fair. Always cite relevant regulations.
Focus on actual safety risks, not minor technicalities."""

        log.info("policy_agent_initialized", strict_mode=strict_mode)

    def _get_ppe_requirements(self, zone: SafetyZone) -> dict[str, str]:
        """Get PPE requirements with OSHA references."""
        requirements = {}

        ppe_osha_map = {
            "hardhat": "29 CFR 1910.135",
            "safety helmet": "29 CFR 1910.135",
            "safety glasses": "29 CFR 1910.133",
            "goggles": "29 CFR 1910.133",
            "face shield": "29 CFR 1910.133",
            "welding mask": "29 CFR 1910.252",
            "gloves": "29 CFR 1910.138",
            "safety gloves": "29 CFR 1910.138",
            "safety vest": "29 CFR 1910.132",
            "high visibility vest": "29 CFR 1910.132",
            "respirator": "29 CFR 1910.134",
            "hearing protection": "29 CFR 1910.95"
        }

        for ppe in zone.required_ppe:
            ppe_lower = ppe.lower()
            ref = ppe_osha_map.get(ppe_lower, "29 CFR 1910.132")
            requirements[ppe] = ref

        return requirements

    def _normalize_ppe_name(self, name: str) -> str:
        """Normalize PPE names for comparison."""
        name_lower = name.lower().replace("_", " ")

        # Normalize common variations
        mappings = {
            "hard hat": "hardhat",
            "safety helmet": "hardhat",
            "helmet": "hardhat",
            "vest": "safety vest",
            "high vis": "safety vest",
            "glasses": "safety glasses",
            "goggles": "safety glasses",
            "glove": "gloves",
            "face shield": "face shield",
            "welding helmet": "welding mask"
        }

        for pattern, normalized in mappings.items():
            if pattern in name_lower:
                return normalized

        return name_lower

    def _check_ppe_compliance(
        self,
        scene: SceneDescription,
        zone: SafetyZone
    ) -> tuple[list[str], list[str]]:
        """
        Check PPE compliance for all workers.

        Returns:
            Tuple of (missing_items, osha_references)
        """
        if not zone.required_ppe:
            return [], []

        requirements = self._get_ppe_requirements(zone)
        missing = []
        references = set()

        for worker_id, worker_ppe in scene.ppe_status.items():
            # Normalize worker's PPE
            worker_ppe_normalized = [self._normalize_ppe_name(p) for p in worker_ppe]

            for required_ppe, osha_ref in requirements.items():
                required_normalized = self._normalize_ppe_name(required_ppe)

                if required_normalized not in worker_ppe_normalized:
                    if required_ppe not in missing:
                        missing.append(required_ppe)
                    references.add(osha_ref)

        return missing, list(references)

    def _determine_severity(
        self,
        violation_type: str,
        zone: SafetyZone
    ) -> str:
        """Determine severity based on violation and zone type."""
        if violation_type == "restricted_area":
            return "critical"

        if violation_type == "overcrowded":
            return "warning"

        if violation_type == "missing_ppe":
            severity_map = VIOLATION_SEVERITY.get("missing_ppe", {})
            return severity_map.get(zone.zone_type, "warning")

        return "warning"

    def _generate_explanation(
        self,
        violation_type: str,
        missing_items: list[str],
        zone: SafetyZone,
        references: list[str]
    ) -> str:
        """Generate human-readable explanation."""
        explanations = []

        if violation_type == "missing_ppe":
            items_str = ", ".join(missing_items)
            explanations.append(
                f"Worker(s) in {zone.name} missing required PPE: {items_str}."
            )
            explanations.append(
                f"Zone type '{zone.zone_type.value}' requires specific protective equipment."
            )

        elif violation_type == "restricted_area":
            explanations.append(
                f"Unauthorized entry detected in restricted zone: {zone.name}."
            )
            explanations.append(
                "Only authorized personnel with proper clearance may enter."
            )

        elif violation_type == "overcrowded":
            explanations.append(
                f"Zone {zone.name} has exceeded maximum occupancy of {zone.max_occupancy}."
            )

        if references:
            ref_str = ", ".join(references)
            explanations.append(f"Relevant OSHA standards: {ref_str}")

        return " ".join(explanations)

    def _generate_recommendations(
        self,
        violation_type: str,
        missing_items: list[str],
        zone: SafetyZone
    ) -> list[str]:
        """Generate recommendations to resolve violation."""
        recommendations = []

        if violation_type == "missing_ppe":
            for item in missing_items:
                recommendations.append(f"Obtain and wear {item} before entering {zone.name}")

            if zone.osha_reference:
                recommendations.append(f"Review requirements in {zone.osha_reference}")

        elif violation_type == "restricted_area":
            recommendations.append("Exit restricted area immediately")
            recommendations.append("Contact supervisor for authorization if access is required")

        elif violation_type == "overcrowded":
            recommendations.append("Some workers should exit zone until below capacity")
            recommendations.append(f"Maximum occupancy is {zone.max_occupancy} workers")

        return recommendations

    async def evaluate(
        self,
        scene: SceneDescription,
        zone: SafetyZone
    ) -> PolicyDecision:
        """
        Evaluate scene against zone policies.

        Args:
            scene: Scene description from perception agent
            zone: Zone to check against

        Returns:
            PolicyDecision with evaluation results
        """
        start_time = time.time()

        # Check for various violation types
        violation_type = None
        missing_items = []
        osha_references = []
        severity = "info"

        # 1. Check restricted area
        if zone.zone_type == ZoneType.RESTRICTED and scene.workers_detected > 0:
            violation_type = "restricted_area"
            severity = "critical"
            osha_references = ["29 CFR 1910.147"]  # General duty clause

        # 2. Check PPE compliance
        if zone.zone_type in (ZoneType.PPE_REQUIRED, ZoneType.HAZARD, ZoneType.MACHINE):
            missing, refs = self._check_ppe_compliance(scene, zone)
            if missing:
                violation_type = violation_type or "missing_ppe"
                missing_items = missing
                osha_references.extend(refs)
                severity = self._determine_severity("missing_ppe", zone)

        # 3. Check occupancy
        if zone.max_occupancy and scene.workers_detected > zone.max_occupancy:
            if not violation_type:  # Don't override more severe violations
                violation_type = "overcrowded"
                severity = "warning"

        # 4. Consider safety concerns from perception
        if self.strict_mode and scene.safety_concerns and not violation_type:
            # Flag potential issues from VLM analysis
            for concern in scene.safety_concerns:
                if "missing" in concern.lower() or "without" in concern.lower():
                    violation_type = "missing_ppe"
                    severity = "info"
                    break

        # Generate decision
        is_violation = violation_type is not None

        explanation = ""
        recommendations = []

        if is_violation:
            explanation = self._generate_explanation(
                violation_type, missing_items, zone, osha_references
            )
            recommendations = self._generate_recommendations(
                violation_type, missing_items, zone
            )

            # Add zone's OSHA reference if available
            if zone.osha_reference and zone.osha_reference not in osha_references:
                osha_references.append(zone.osha_reference)

        eval_time = (time.time() - start_time) * 1000

        decision = PolicyDecision(
            is_violation=is_violation,
            violation_type=violation_type,
            severity=severity,
            missing_items=missing_items,
            osha_references=osha_references,
            explanation=explanation,
            confidence=scene.confidence,
            evaluation_time_ms=eval_time,
            recommendations=recommendations
        )

        log.info(
            "policy_evaluation_complete",
            is_violation=is_violation,
            type=violation_type,
            severity=severity,
            time_ms=round(eval_time, 1)
        )

        return decision

    def get_info(self) -> dict:
        """Get agent information."""
        return {
            "name": "PolicyAgent",
            "strict_mode": self.strict_mode,
            "has_rag": self._rag is not None
        }
