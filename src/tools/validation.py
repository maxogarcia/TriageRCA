"""Shared input checks. All tool windows use Unix seconds."""
import math


def validate_query(case, service, start_time, end_time):
    for value in (start_time, end_time):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Window bounds must be finite Unix seconds")
    if start_time >= end_time:
        raise ValueError("start_time must be less than end_time")
    if max(abs(start_time), abs(end_time)) >= 1e11:
        raise ValueError("Use Unix seconds, not milliseconds/microseconds/nanoseconds")
    if service not in case.services:
        raise ValueError(f"Unknown service {service!r}. Valid services: {', '.join(case.services)}")
