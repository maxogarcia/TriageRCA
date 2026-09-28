"""Service name normalization across telemetry sources.

Confirmed on RE2-OB (Online Boutique) by inspecting real RCAEval parquet files: metrics and logs
name it "frontend", traces name the same service "frontendservice". Other services matched
exactly. Other RCAEval systems (SockShop, TrainTicket) may have their own quirks -- add aliases
here as they're found, rather than hardcoding lookups in individual tools.
"""

from __future__ import annotations

# canonical (metrics/logs) name -> name used in traces.serviceName
_CANONICAL_TO_TRACE_NAME: dict[str, str] = {
    "frontend": "frontendservice",
}


def trace_service_names(canonical_service: str) -> set[str]:
    """All names a service might appear under in traces.serviceName, given its canonical
    (metrics/logs) name. Always includes the canonical name itself as a fallback."""
    alias = _CANONICAL_TO_TRACE_NAME.get(canonical_service)
    return {canonical_service, alias} if alias else {canonical_service}
