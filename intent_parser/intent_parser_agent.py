
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()  # .env içindeki GROQ_API_KEY'i ortam değişkeni olarak yükler

# --- 1) Şema: Sonraki agent'ların kullanacağı yapı ---

KpiName = Literal["RX_POWER", "SINR", "THROUGHPUT_5P", "SERVED_USERS"]
Op = Literal["GT", "GTE", "LT", "LTE", "BETWEEN", "DELTA_UP", "DELTA_DOWN", "TARGET"]

class KpiThreshold(BaseModel):
    kpi: KpiName
    op: Op = Field(..., description="Comparison/intent operator")
    value: Optional[float] = Field(None, description="Single threshold or target value")
    value_low: Optional[float] = Field(None, description="Lower bound for BETWEEN")
    value_high: Optional[float] = Field(None, description="Upper bound for BETWEEN")
    delta: Optional[float] = Field(None, description="Magnitude for DELTA_UP / DELTA_DOWN")
    unit: Optional[str] = Field(None, description="e.g., dBm, dB, Mbps, users")

class IntentParse(BaseModel):
    """User's intent - what they want to achieve (not how to achieve it)"""
    target_area: str = Field(..., description="Where to apply: site/cell/cluster/city/coordinates")
    target_kpis: List[KpiName] = Field(..., min_length=1)
    kpi_thresholds: List[KpiThreshold] = Field(default_factory=list)
    time_constraint_start: Optional[str] = Field(None, description="ISO 8601 format: 2026-02-05T10:00:00")
    time_constraint_end: Optional[str] = Field(None, description="ISO 8601 format: 2026-02-05T14:00:00")
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    confidence: Optional[float] = Field(default=0.8, ge=0, le=1, description="Confidence score (auto-generated)")


INSTRUCTIONS = [
    "You are an expert intent parser for a cellular/mobile network optimization system.",
    "Your task: Parse natural language requests into structured JSON following the IntentParse schema.",
    "",
    "## KPI Mapping Rules:",
    "- Coverage/Signal Strength/RX/Reception → RX_POWER",
    "- Quality/Interference/SINR/Signal Quality → SINR",
    "- Throughput/Speed/TP/5-percentile/P5 → THROUGHPUT_5P",
    "- Load/Density/User Count/Balance → SERVED_USERS",
    "",
    "## Operator Mapping:",
    "- 'at least', '>=', 'above or equal' → GTE",
    "- 'above', '>', 'more than' → GT",
    "- 'at most', '<=', 'below or equal' → LTE",
    "- 'below', '<', 'less than' → LT",
    "- 'between X and Y' → BETWEEN (use value_low and value_high)",
    "- 'increase by', 'boost by' → DELTA_UP (use delta field)",
    "- 'decrease by', 'reduce by' → DELTA_DOWN (use delta field)",
    "- 'target', 'aim for', 'balance', 'optimize' → TARGET",
    "",
    "## Priority Inference:",
    "- CRITICAL: Contains 'critical', 'urgent', 'emergency', or severe issues",
    "- HIGH: Contains 'high priority', 'important', 'asap'",
    "- MEDIUM: Default if not specified",
    "- LOW: Contains 'low priority', 'when possible', 'minor'",
    "",
    "## Confidence Scoring:",
    "- 0.95-1.0: All fields clear (area, KPI, threshold, optional time)",
    "- 0.85-0.94: Area + KPI + threshold clear, some optional fields missing",
    "- 0.70-0.84: One key field ambiguous (e.g., vague area or no threshold)",
    "- 0.50-0.69: Two key fields ambiguous or missing",
    "- Below 0.50: Multiple critical fields unclear or missing",
    "",
    "## Important Rules:",
    "- Focus on WHAT the user wants (goals), not HOW to achieve it (configuration)",
    "- Always extract target_area (city, site ID, cell ID, or coordinates)",
    "- Extract ALL mentioned KPIs into target_kpis array",
    "- Parse time constraints into ISO 8601 format (YYYY-MM-DDTHH:MM:SS)",
    "- Be precise with units (dBm, dB, Mbps, users, degrees, etc.)",
]

intent_parser_agent = Agent(
    name="Intent Parser",
    description="Parses natural language network intents into structured features for downstream agents.",
    model=Groq(id="llama-3.3-70b-versatile"),  # Llama 3.3 70B versatile
    output_schema=IntentParse,
    instructions=INSTRUCTIONS,
    markdown=True,  # Better instruction parsing
    structured_outputs=True,  # Enforce schema compliance
)

if __name__ == "__main__":
    examples = [
        "Improve coverage in Kadıköy area. RX power should be at least -95 dBm. High priority. This evening between 18:00-23:00.",
        "Quality is poor in site TR-IST-034 cell, SINR should be above 10 dB. Increase tilt by 2 degrees if necessary.",
        "Load is unbalanced in Ankara Çankaya; balance the number of served users, critical.",
        "Speed is very low in Beşiktaş, 5-percentile throughput should be at least 8 Mbps.",
    ]
    for q in examples:
        out = intent_parser_agent.run(q)
        print("\n--- INPUT ---")
        print(q)
        print("--- OUTPUT ---")
        print(out.content)