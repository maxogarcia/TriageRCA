from __future__ import annotations

from src.llm.cache import _is_rate_limit_error


class _ApiError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def test_per_minute_quota_and_overload_are_retried():
    assert _is_rate_limit_error(_ApiError(429, "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))
    assert _is_rate_limit_error(_ApiError(503, "This model is currently experiencing high demand."))


def test_per_day_quota_is_not_retried():
    assert not _is_rate_limit_error(_ApiError(429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
