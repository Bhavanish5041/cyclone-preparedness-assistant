"""Experimental Assistant Adapter for CycloneLM.

Connects CycloneLMEngine and ToolOrchestrator into a unified conversational service.
Maintains experimental isolation from existing production FastAPI routes while
providing a drop-in interface for future integration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.cyclonelm.inference.engine import CycloneLMEngine
from src.cyclonelm.tools.protocol import ToolOrchestrator

logger = logging.getLogger(__name__)

SCIENTIFIC_PRECEDENCE_NOTE = (
    "AHP PRECEDENCE MANDATE: Deterministic AHP Preparedness Scores and 4-band priority classifications "
    "maintain absolute operational precedence for disaster preparedness and emergency resource allocation. "
    "Logistic Regression class probabilities represent an empirical tier surrogate approximating AHP classification boundaries; "
    "they are uncalibrated class likelihoods and do NOT represent physical cyclone-risk or casualty forecasts."
)


class CycloneLMAssistantAdapter:
    """Experimental assistant adapter coordinating CycloneLM generation and deterministic tools."""

    def __init__(
        self,
        engine: Optional[CycloneLMEngine] = None,
        orchestrator: Optional[ToolOrchestrator] = None,
        checkpoint_path: Optional[str | Path] = None,
        tokenizer_path: Optional[str | Path] = None,
    ) -> None:
        self.engine = engine or CycloneLMEngine(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
        )
        self.orchestrator = orchestrator or ToolOrchestrator()

    def process_query(
        self,
        query: str,
        max_tool_rounds: int = 2,
        temperature: float = 0.7,
        mock_tool_call: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process user query through tool orchestration and CycloneLM generation.

        Args:
            query: User's natural language question or instruction.
            max_tool_rounds: Maximum tool execution iterations.
            temperature: Sampling temperature for generation.
            mock_tool_call: Optional simulated tool call string for controlled protocol evaluation.
        """
        # 1. Check if mock call is provided or prompt triggers a tool
        evidence_context = ""
        orchestration_result = None

        if mock_tool_call:
            orchestration_result = self.orchestrator.run_session(
                prompt=query,
                mock_generator_outputs=[mock_tool_call],
            )
            evidence_context = orchestration_result.format_evidence_context()

        # 2. Formulate generation prompt with evidence grounding
        synthesis_prompt = f"<system>{SCIENTIFIC_PRECEDENCE_NOTE}</system>\n"
        if evidence_context:
            synthesis_prompt += f"<evidence>\n{evidence_context}\n</evidence>\n"
        synthesis_prompt += f"<user>{query}</user>\n<assistant>"

        # 3. Model generation (if initialized)
        generation_output: Dict[str, Any] = {}
        if self.engine.model is not None:
            generation_output = self.engine.generate(
                prompt=synthesis_prompt,
                max_new_tokens=60,
                temperature=temperature,
            )

        return {
            "query": query,
            "orchestration": orchestration_result.records if orchestration_result else [],
            "evidence_context": evidence_context,
            "generation": generation_output,
            "governance_notice": SCIENTIFIC_PRECEDENCE_NOTE,
        }

    def health(self) -> Dict[str, Any]:
        """Health check for experimental adapter."""
        return {
            "adapter": "CycloneLMAssistantAdapter",
            "mode": "experimental",
            "production_default": False,
            "engine": self.engine.health(),
            "tools_registered": self.orchestrator.registry.list_tools(),
        }
