from __future__ import annotations

import json
from typing import Any

from ..schemas import ConflictReport, KPIPrediction, OptimizationResult, ResolutionResult, StrategicNarrative
from .runtime import build_groq_agent, coerce_response_model

REASONING_INSTRUCTIONS = [
    "You are the Final Reasoning Agent for a cellular-network operator.",
    "Explain the final decision using only supplied records.",
    "Do not expose hidden chain-of-thought.",
    "Do not invent KPI values or claim simulator validation.",
    "Clearly label all KPI values as surrogate predictions.",
    "Mention hard constraints, conflicts, resolution policy, and unmet targets.",
    "Return only the StrategicNarrative schema.",
]


def create_reasoning_agent(model_id: str) -> Any:
    return build_groq_agent(
        model_id=model_id,
        name="Reasoning Agent",
        role="Produces the final operator-facing engineering explanation.",
        instructions=REASONING_INSTRUCTIONS,
        output_schema=StrategicNarrative,
        structured_outputs=True,
        markdown=False,
        retries=2,
    )


class ReasoningNarrator:
    def __init__(self, model_id: str | None = None, use_llm: bool = True):
        self.model_id = model_id
        self.use_llm = use_llm

    @staticmethod
    def _fallback(
        optimization: OptimizationResult,
        conflict: ConflictReport,
        resolution: ResolutionResult,
        final_prediction: KPIPrediction,
    ) -> StrategicNarrative:
        target_status = "met" if optimization.target_satisfied else "not fully met"
        conflict_text = (
            f"{len(conflict.details)} conflict predicates were triggered."
            if conflict.conflict_detected
            else "No overlapping conflict was detected."
        )
        safety_notes = list(optimization.hard_constraints.notes)
        if not optimization.target_satisfied:
            safety_notes.append(
                "The best surrogate-evaluated candidate did not satisfy every numeric target; review before deployment."
            )
        safety_notes.append(
            "All KPI values are surrogate predictions until confirmed in the DeepMIMO/Colab evaluation notebook."
        )
        return StrategicNarrative(
            summary=(
                f"The Optimization Agent evaluated {len(optimization.iterations)} candidate configuration(s); "
                f"the requested numeric target was {target_status}. The final strategy was {resolution.strategy.value}."
            ),
            optimization_explanation=(
                f"The selected candidate achieved score {optimization.best_score:.4f}. "
                f"Predicted coverage={final_prediction.coverage_ratio}, RX-power p5={final_prediction.rx_power_p5_dbm}, "
                f"SINR p5={final_prediction.sinr_p5_db}, total TX power={final_prediction.total_tx_power_watt:.3f} W."
            ),
            conflict_explanation=conflict_text,
            resolution_explanation=resolution.rationale,
            safety_notes=safety_notes,
        )

    def explain(
        self,
        optimization: OptimizationResult,
        conflict: ConflictReport,
        resolution: ResolutionResult,
        final_prediction: KPIPrediction,
    ) -> StrategicNarrative:
        if not self.use_llm or not self.model_id:
            return self._fallback(optimization, conflict, resolution, final_prediction)
        payload = {
            "optimization": optimization.model_dump(mode="json"),
            "conflict_report": conflict.model_dump(mode="json"),
            "resolution": resolution.model_dump(mode="json"),
            "final_surrogate_prediction": final_prediction.model_dump(mode="json"),
        }
        try:
            response = create_reasoning_agent(self.model_id).run(
                "Create the final operator-facing narrative from this JSON record:\n"
                + json.dumps(payload, indent=2)
            )
            return coerce_response_model(response, StrategicNarrative)
        except Exception:
            return self._fallback(optimization, conflict, resolution, final_prediction)
