"""log_search tool: keyword/level-filtered, top-N capped log lines for a service+window.

`level` is honored only when the source actually has a level column -- RCAEval's RE2-OB logs
(read_logs()) come back as just [timestamp, container_name, message], no level/error field, so a
hardcoded df["level"] lookup would KeyError on that system. Other RCAEval systems (SockShop,
TrainTicket) may include it, so this checks df.columns rather than assuming either way.
"""

from __future__ import annotations

from src.data.case import Case
from src.llm.base import ToolSpec
from src.tools.validation import validate_query

SPEC = ToolSpec(
    name="log_search",
    description="Search a service's logs in a time window, optionally filtered by level or keyword.",
    parameters={
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "Container/service name to filter on"},
            "start_time": {"type": "number"},
            "end_time": {"type": "number"},
            "level": {"type": "string", "description": "Optional log level filter, e.g. 'error'"},
            "keyword": {
                "type": "string",
                "description": "Optional substring to filter message text on",
            },
        },
        "required": ["service", "start_time", "end_time"],
    },
)

_MAX_LINES = 20


def make_log_search(case: Case):
    def log_search(
        service: str,
        start_time: float,
        end_time: float,
        level: str | None = None,
        keyword: str | None = None,
    ) -> str:
        validate_query(case, service, start_time, end_time)
        df = case.logs
        if df is None or df.empty:
            return "Log source unavailable or empty; this is not evidence of normal behavior."
        mask = (
            (df["timestamp"] >= start_time)
            & (df["timestamp"] <= end_time)
            & df["container_name"].astype("string").str.contains(service, case=False, na=False, regex=False)
        )
        has_level = "level" in df.columns
        if level and has_level:
            mask &= df["level"].astype("string").str.casefold() == level.casefold()
        elif level and not has_level:
            # Fall back to treating the requested level as a message keyword, since this
            # source has no structured level field.
            mask &= df["message"].astype("string").str.contains(level, case=False, na=False, regex=False)
        if keyword:
            mask &= df["message"].astype("string").str.contains(keyword, case=False, na=False, regex=False)

        matched = df[mask]
        if matched.empty:
            return f"No matching observations for service='{service}' level={level!r} keyword={keyword!r}; not proof of health."

        lines = [f"{len(matched)} matching log lines (showing up to {_MAX_LINES}):"]
        lines.append("Container matching uses a literal substring, not a verified exact service identity.")
        if level and not has_level:
            lines.append("No structured level field; level filter was applied as a message keyword.")
        for _, row in matched.head(_MAX_LINES).iterrows():
            level_str = f"{row['level']}: " if has_level else ""
            lines.append(f"  [{row['timestamp']:.0f}] {level_str}{row['message'][:200]}")
        if len(matched) > _MAX_LINES:
            lines.append(f"  ... ({len(matched) - _MAX_LINES} more lines omitted)")
        return "\n".join(lines)

    return log_search
