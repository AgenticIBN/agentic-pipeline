from __future__ import annotations

from typing import Any

from ..schemas import IntentParse
from .runtime import build_groq_agent, coerce_response_model

INSTRUCTIONS = [
    "You are the Intent Parser Agent for a cellular-network intent-based networking system.",
    "Convert the user's request into the IntentParse output schema.",
    "Extract goals only. Do not propose transmitter powers, beam angles, or a final configuration.",
    "Always identify a target_area. Use 'global' only when the request truly applies everywhere.",
    "Map coverage or reception requests to RX_POWER and/or COVERAGE as appropriate.",
    "Map interference or signal-quality requests to SINR.",
    "Map speed or fifth-percentile throughput requests to THROUGHPUT_5P.",
    "Map energy or power-consumption requests to TOTAL_TX_POWER.",
    "Map user-distribution requests to LOAD_BALANCE, LOAD_IMBALANCE, or SERVED_USERS.",
    "For explicit TX ON/OFF or maintenance requests, include COVERAGE as an objective because service continuity must be checked.",
    "Use GT, GTE, LT, LTE, BETWEEN, DELTA_UP, DELTA_DOWN, or TARGET for threshold operators.",
    "For open-ended improve/maximize/minimize requests, use TARGET with no numeric value.",
    "Infer priority as CRITICAL, HIGH, MEDIUM, or LOW; MEDIUM is the default.",
    "Use ISO-8601 datetimes when a complete time can be resolved; otherwise leave the field empty.",
    "Return only schema-compliant structured data.",
]


def create_intent_parser_agent(model_id: str) -> Any:
    return build_groq_agent(
        model_id=model_id,
        name="Intent Parser Agent",
        role="Transforms natural-language network intents into typed goals and constraints.",
        description="Semantic entry point for the conflict-aware IBN workflow.",
        instructions=INSTRUCTIONS,
        output_schema=IntentParse,
        structured_outputs=True,
        markdown=False,
        retries=2,
    )


def run_intent_parser(intent_text: str, model_id: str) -> IntentParse:
    agent = create_intent_parser_agent(model_id)
    response = agent.run(intent_text)
    return coerce_response_model(response, IntentParse)
