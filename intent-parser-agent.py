
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.google import Gemini  
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

class ConfigChange(BaseModel):
    action: str = Field(..., description="Free-text config action, e.g., 'increase tilt by 2 degrees'")
    parameter: Optional[str] = Field(None, description="Parameter name if explicitly mentioned")
    direction: Optional[Literal["INCREASE", "DECREASE", "SET", "OPTIMIZE"]] = None
    amount: Optional[float] = None
    unit: Optional[str] = None

class IntentParse(BaseModel):
    target_area: str = Field(..., description="Where to apply: site/cell/cluster/city/coordinates")
    target_kpis: List[KpiName] = Field(..., min_length=1)
    kpi_thresholds: List[KpiThreshold] = Field(default_factory=list)
    time_constraint_start: Optional[str] = Field(None)
    time_constraint_end: Optional[str] = Field(None)
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    configuration_change: List[ConfigChange] = Field(default_factory=list)
    affected_sectors: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)


INSTRUCTIONS = [
    "You are an intent parser for a cellular/mobile network optimization system.",
    "Convert the user's natural language request into the provided schema ONLY.",
    "Map KPI mentions using this dictionary:",
    "- coverage area / coverage / signal strength / rx -> RX_POWER",
    "- quality / quality / interference / sinr -> SINR",
    "- throughput / tp / speed / 5-percentile / p5 -> THROUGHPUT_5P",
    "- load / density / number of users / balance -> SERVED_USERS",
    "If the user asks to 'increase/decrease' without a number, set a DELTA_UP/DOWN with delta=None and keep unit if known.",
    "If the user gives a target like 'SINR >= 10 dB' use op=GTE and value=10 unit='dB'.",
    "If there is no explicit config change, leave configuration_change empty.",
    "Only fill affected_sectors if explicitly stated (e.g., 'sector A', 'cell 3', 'azimuth 120').",
    "Estimate confidence: 0.9+ when area+KPI+threshold are clear; 0.6-0.8 when one piece is vague; <0.6 when multiple are missing.",
]

intent_parser_agent = Agent(
    name="Intent Parser",
    description="Parses natural language network intents into structured features for downstream agents.",
    model=Gemini(id="gemini-1.5-flash"),  
    output_schema=IntentParse,
    instructions=INSTRUCTIONS,
)

if __name__ == "__main__":
    examples = [
        "Kadıköy bölgesinde coverage'ı artır. RX power en az -95 dBm olsun. Öncelik yüksek. Bu akşam 18:00-23:00 arası.",
        "Site TR-IST-034 hücresinde kalite kötü, SINR 10 dB üstüne çıksın. Gerekirse tilt 2 derece artır.",
        "Ankara Çankaya'da yük dengesiz; serve edilen kullanıcı sayısı dengelensin, kritik.",
        "Beşiktaş'ta hız çok düşük, 5-percentile throughput en az 8 Mbps olsun.",
    ]
    for q in examples:
        out = intent_parser_agent.run(q)
        print("\n--- INPUT ---")
        print(q)
        print("--- OUTPUT ---")
        print(out.content)