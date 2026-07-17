from __future__ import annotations

import re

from .schemas import HardConstraints


def extract_hard_constraints(intent_text: str) -> HardConstraints:
    text = intent_text or ""
    tx_states: dict[int, bool] = {}
    notes: list[str] = []

    off_patterns = [
        r"(?:turn\s+off|deactivate|shut\s*down|disable|switch\s+off)\s+(?:the\s+)?(?:base\s+station\s+|transmitter\s+)?tx\s*([0-3])",
        r"tx\s*([0-3]).{0,30}(?:maintenance|outage|must\s+be\s+off)",
        r"tx\s*([0-3])_on\s*(?:=|:|to)\s*(?:false|off|0)",
    ]
    on_patterns = [
        r"(?:turn\s+on|activate|enable|switch\s+on)\s+(?:the\s+)?(?:base\s+station\s+|transmitter\s+)?tx\s*([0-3])",
        r"tx\s*([0-3])_on\s*(?:=|:|to)\s*(?:true|on|1)",
    ]

    for pattern in off_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            index = int(match.group(1))
            tx_states[index] = False
            notes.append(f"TX{index} must remain OFF because the original intent contains a shutdown or maintenance command.")

    for pattern in on_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            index = int(match.group(1))
            tx_states[index] = True
            notes.append(f"TX{index} must remain ON because the original intent contains an activation command.")

    all_available = bool(
        re.search(
            r"\b(?:all\s+available\s+(?:base\s+stations|transmitters|txs?)|all\s+(?:base\s+stations|transmitters|txs?))\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if all_available:
        for index in range(4):
            tx_states.setdefault(index, True)
        notes.append("All available base stations were requested; every TX without a contradictory explicit shutdown is constrained ON.")

    return HardConstraints(tx_states=tx_states, all_available_requested=all_available, notes=notes)
