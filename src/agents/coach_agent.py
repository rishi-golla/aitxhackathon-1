"""
Coach Agent for generating constructive safety feedback.

Creates friendly, helpful messages for workers - copilot, not cop.
"""

import random
import time
from dataclasses import dataclass
from typing import Optional
import structlog

from src.agents.policy_agent import PolicyDecision
from src.zones.zone_manager import SafetyZone, ZoneType
from src.tts.alert_system import AlertPriority

log = structlog.get_logger()


@dataclass
class CoachFeedback:
    """Feedback from coach agent."""
    message: str  # Human-friendly, constructive message
    tone: str  # "reminder", "instruction", "urgent"
    addressed_to: str  # "worker in zone A" or specific ID
    include_why: bool  # Explain the regulation
    audio_priority: AlertPriority
    short_message: str  # Brief version for TTS
    full_explanation: Optional[str] = None


# Message templates by violation type and tone
MESSAGE_TEMPLATES = {
    "missing_ppe": {
        "reminder": [
            "Hey! Don't forget your {item} before heading into {zone}.",
            "Quick reminder - {item} needed in {zone}!",
            "Looking good! Just need your {item} for {zone}.",
            "Safety check: {item} required for {zone}."
        ],
        "instruction": [
            "Please grab your {item} before continuing in {zone}.",
            "Hold up! {zone} requires {item}. Please equip it now.",
            "You'll need your {item} to work in {zone} safely.",
            "Safety first! Get your {item} before entering {zone}."
        ],
        "urgent": [
            "Stop! {item} required immediately in {zone}!",
            "Attention! Exit {zone} and get your {item} now.",
            "Safety alert! Missing {item} in hazardous zone. Equip immediately."
        ]
    },
    "restricted_area": {
        "reminder": [
            "Heads up - you're near a restricted area.",
            "This zone requires authorization.",
        ],
        "instruction": [
            "Please step back - this area is restricted.",
            "Authorization needed for this zone. Check with your supervisor.",
        ],
        "urgent": [
            "Warning! Restricted area. Please exit immediately.",
            "Stop! Restricted zone - authorized personnel only.",
        ]
    },
    "overcrowded": {
        "reminder": [
            "Zone {zone} is getting full. Please be aware.",
            "We're at capacity in {zone}.",
        ],
        "instruction": [
            "Zone {zone} is at capacity. Please wait for clearance.",
            "Too many workers in {zone}. Some should exit before continuing.",
        ],
        "urgent": [
            "Zone {zone} overcrowded! Some workers must exit now.",
        ]
    }
}

# Why explanations
WHY_EXPLANATIONS = {
    "hardhat": "Hard hats protect against falling objects and head injuries.",
    "safety glasses": "Safety glasses prevent eye injuries from debris and particles.",
    "safety vest": "High-visibility vests help you be seen by equipment operators.",
    "gloves": "Gloves protect your hands from cuts, burns, and chemical exposure.",
    "face shield": "Face shields provide additional protection for your face from splashes and sparks.",
    "welding mask": "Welding masks protect your eyes from harmful UV/IR radiation and bright arc light.",
    "respirator": "Respirators filter harmful particles and fumes from the air you breathe.",
    "restricted_area": "Restricted areas contain hazards that require special training and authorization.",
    "overcrowded": "Limiting zone occupancy prevents accidents and ensures safe evacuation routes."
}


class CoachAgent:
    """
    Coach agent that generates constructive safety feedback.

    Philosophy: Be a safety copilot, not a cop.
    - Friendly and helpful, not punitive
    - Brief and actionable
    - Explain the 'why' when helpful
    - Respectful of skilled workers
    """

    def __init__(
        self,
        include_explanations: bool = True,
        use_names: bool = True
    ):
        """
        Initialize coach agent.

        Args:
            include_explanations: Include 'why' explanations
            use_names: Use worker IDs when available
        """
        self.include_explanations = include_explanations
        self.use_names = use_names

        self._system_prompt = """You are a friendly safety coach, NOT a cop.
Your job is to help workers stay safe, not punish them.

Guidelines:
- Be constructive: "Hey, grab your hardhat!" not "VIOLATION DETECTED"
- Be brief: Workers are busy
- Be helpful: Explain WHY the PPE matters
- Be respectful: These are skilled professionals"""

        log.info("coach_agent_initialized")

    def _determine_tone(self, decision: PolicyDecision) -> str:
        """Determine message tone based on severity."""
        if decision.severity == "critical":
            return "urgent"
        elif decision.severity == "warning":
            return "instruction"
        else:
            return "reminder"

    def _get_priority(self, decision: PolicyDecision) -> AlertPriority:
        """Map severity to alert priority."""
        if decision.severity == "critical":
            return AlertPriority.CRITICAL
        elif decision.severity == "warning":
            return AlertPriority.WARNING
        else:
            return AlertPriority.INFO

    def _format_items(self, items: list[str]) -> str:
        """Format list of items for natural language."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _select_template(
        self,
        violation_type: str,
        tone: str
    ) -> str:
        """Select a message template."""
        templates = MESSAGE_TEMPLATES.get(violation_type, {}).get(tone, [])
        if templates:
            return random.choice(templates)

        # Fallback
        return "Safety reminder for this zone. Please check compliance."

    def _generate_explanation(
        self,
        violation_type: str,
        missing_items: list[str]
    ) -> str:
        """Generate explanation for why PPE is needed."""
        explanations = []

        if violation_type == "missing_ppe":
            for item in missing_items:
                item_lower = item.lower()
                for key, explanation in WHY_EXPLANATIONS.items():
                    if key in item_lower:
                        explanations.append(explanation)
                        break

        elif violation_type in WHY_EXPLANATIONS:
            explanations.append(WHY_EXPLANATIONS[violation_type])

        return " ".join(explanations)

    async def generate_feedback(
        self,
        decision: PolicyDecision,
        zone: SafetyZone,
        worker_id: Optional[str] = None
    ) -> CoachFeedback:
        """
        Generate friendly feedback for a policy decision.

        Args:
            decision: Policy decision from policy agent
            zone: Zone where violation occurred
            worker_id: Optional worker identifier

        Returns:
            CoachFeedback with constructive message
        """
        start_time = time.time()

        if not decision.is_violation:
            # No violation - optional positive feedback
            return CoachFeedback(
                message="Looking good! Keep up the safe work!",
                tone="reminder",
                addressed_to=f"worker in {zone.name}",
                include_why=False,
                audio_priority=AlertPriority.INFO,
                short_message="Looking good!"
            )

        # Determine tone and priority
        tone = self._determine_tone(decision)
        priority = self._get_priority(decision)

        # Format items for message
        items_str = self._format_items(decision.missing_items)

        # Select and format template
        template = self._select_template(decision.violation_type, tone)

        # Replace placeholders
        message = template.format(
            item=items_str or "required PPE",
            zone=zone.name,
            items=items_str
        )

        # Generate short version for TTS
        if decision.violation_type == "missing_ppe" and decision.missing_items:
            short_message = f"{items_str} needed in {zone.name}!"
        elif decision.violation_type == "restricted_area":
            short_message = "Restricted area - please exit."
        elif decision.violation_type == "overcrowded":
            short_message = f"{zone.name} at capacity."
        else:
            short_message = "Safety alert - please check compliance."

        # Generate explanation if needed
        full_explanation = None
        if self.include_explanations:
            explanation = self._generate_explanation(
                decision.violation_type,
                decision.missing_items
            )
            if explanation:
                full_explanation = explanation

                # Add OSHA reference
                if decision.osha_references:
                    full_explanation += f" (Reference: {decision.osha_references[0]})"

        # Determine who to address
        if worker_id and self.use_names:
            addressed_to = f"Worker {worker_id}"
        else:
            addressed_to = f"Worker in {zone.name}"

        gen_time = (time.time() - start_time) * 1000

        feedback = CoachFeedback(
            message=message,
            tone=tone,
            addressed_to=addressed_to,
            include_why=self.include_explanations,
            audio_priority=priority,
            short_message=short_message,
            full_explanation=full_explanation
        )

        log.info(
            "coach_feedback_generated",
            tone=tone,
            priority=priority.name,
            time_ms=round(gen_time, 1)
        )

        return feedback

    def get_info(self) -> dict:
        """Get agent information."""
        return {
            "name": "CoachAgent",
            "include_explanations": self.include_explanations,
            "use_names": self.use_names
        }


# Quick feedback generation for common scenarios
def quick_ppe_reminder(item: str, zone_name: str) -> str:
    """Generate a quick PPE reminder."""
    templates = [
        f"Hey! Don't forget your {item} for {zone_name}!",
        f"Quick reminder: {item} required in {zone_name}.",
        f"Heads up - grab your {item} before entering {zone_name}."
    ]
    return random.choice(templates)


def quick_zone_warning(zone_name: str, zone_type: str) -> str:
    """Generate a quick zone warning."""
    if zone_type == "restricted":
        return f"Warning: {zone_name} is a restricted area."
    elif zone_type == "hazard":
        return f"Caution: {zone_name} is a hazard zone. Full PPE required."
    else:
        return f"Entering {zone_name}. Follow zone safety rules."
