"""
Tests for Agent Framework.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.perception_agent import PerceptionAgent
from src.agents.policy_agent import PolicyAgent, PolicyDecision
from src.agents.coach_agent import CoachAgent, CoachFeedback
from src.agents.safety_chain import SafetyAgentChain
from src.pipeline.vlm_analysis import SceneDescription
from src.zones.zone_manager import SafetyZone, ZoneType


class TestPolicyAgent:
    """Test suite for PolicyAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PolicyAgent()

        self.ppe_zone = SafetyZone(
            zone_id="welding",
            name="Welding Bay",
            zone_type=ZoneType.PPE_REQUIRED,
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            required_ppe=["hardhat", "safety_glasses", "welding_mask"],
            osha_reference="29 CFR 1910.252"
        )

        self.restricted_zone = SafetyZone(
            zone_id="electrical",
            name="Electrical Room",
            zone_type=ZoneType.RESTRICTED,
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)]
        )

    @pytest.mark.asyncio
    async def test_compliant_scene(self):
        """Test evaluation of compliant scene."""
        scene = SceneDescription(
            workers_detected=1,
            activities=["welding"],
            hazards_observed=[],
            ppe_status={"worker_1": ["hardhat", "safety_glasses", "welding_mask"]},
            zone_assessment="Worker properly equipped",
            safety_concerns=[],
            confidence=0.9,
            inference_time_ms=100
        )

        decision = await self.agent.evaluate(scene, self.ppe_zone)

        assert not decision.is_violation
        assert decision.severity == "info"

    @pytest.mark.asyncio
    async def test_missing_ppe_violation(self):
        """Test detection of missing PPE."""
        scene = SceneDescription(
            workers_detected=1,
            activities=["welding"],
            hazards_observed=[],
            ppe_status={"worker_1": ["hardhat"]},  # Missing glasses and mask
            zone_assessment="Worker missing PPE",
            safety_concerns=[],
            confidence=0.9,
            inference_time_ms=100
        )

        decision = await self.agent.evaluate(scene, self.ppe_zone)

        assert decision.is_violation
        assert decision.violation_type == "missing_ppe"
        assert len(decision.missing_items) > 0
        assert len(decision.osha_references) > 0

    @pytest.mark.asyncio
    async def test_restricted_area_violation(self):
        """Test detection of restricted area entry."""
        scene = SceneDescription(
            workers_detected=1,
            activities=["walking"],
            hazards_observed=[],
            ppe_status={},
            zone_assessment="Worker in restricted area",
            safety_concerns=[],
            confidence=0.9,
            inference_time_ms=100
        )

        decision = await self.agent.evaluate(scene, self.restricted_zone)

        assert decision.is_violation
        assert decision.violation_type == "restricted_area"
        assert decision.severity == "critical"


class TestCoachAgent:
    """Test suite for CoachAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = CoachAgent()

        self.zone = SafetyZone(
            zone_id="welding",
            name="Welding Bay",
            zone_type=ZoneType.PPE_REQUIRED,
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            required_ppe=["hardhat"]
        )

    @pytest.mark.asyncio
    async def test_generate_warning_feedback(self):
        """Test generating warning-level feedback."""
        decision = PolicyDecision(
            is_violation=True,
            violation_type="missing_ppe",
            severity="warning",
            missing_items=["hardhat"],
            osha_references=["29 CFR 1910.135"],
            explanation="Missing hardhat in welding area"
        )

        feedback = await self.agent.generate_feedback(decision, self.zone)

        assert feedback.tone == "instruction"
        assert "hardhat" in feedback.message.lower() or "Welding Bay" in feedback.message
        assert feedback.audio_priority.value == 2  # WARNING

    @pytest.mark.asyncio
    async def test_generate_critical_feedback(self):
        """Test generating critical-level feedback."""
        decision = PolicyDecision(
            is_violation=True,
            violation_type="restricted_area",
            severity="critical",
            missing_items=[],
            osha_references=[]
        )

        feedback = await self.agent.generate_feedback(decision, self.zone)

        assert feedback.tone == "urgent"
        assert feedback.audio_priority.value == 3  # CRITICAL

    @pytest.mark.asyncio
    async def test_compliant_feedback(self):
        """Test feedback for compliant scene."""
        decision = PolicyDecision(
            is_violation=False,
            violation_type=None,
            severity="info"
        )

        feedback = await self.agent.generate_feedback(decision, self.zone)

        assert "good" in feedback.message.lower() or "safe" in feedback.message.lower()


class TestSafetyAgentChain:
    """Test suite for SafetyAgentChain."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use mock VLM for testing
        perception = PerceptionAgent(use_mock=True)
        policy = PolicyAgent()
        coach = CoachAgent()

        self.chain = SafetyAgentChain(
            perception=perception,
            policy=policy,
            coach=coach,
            max_chain_time_ms=5000
        )

        self.zone = SafetyZone(
            zone_id="test",
            name="Test Zone",
            zone_type=ZoneType.PPE_REQUIRED,
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            required_ppe=["hardhat"]
        )

    def test_chain_initialization(self):
        """Test chain initializes correctly."""
        assert self.chain.perception is not None
        assert self.chain.policy is not None
        assert self.chain.coach is not None

    def test_get_stats(self):
        """Test getting chain statistics."""
        stats = self.chain.get_stats()

        assert "total_runs" in stats
        assert "violations_detected" in stats
        assert "avg_time_ms" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
