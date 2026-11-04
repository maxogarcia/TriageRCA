# Cross-System Resilience Faults

## GEN-CASCADE-01: Downstream Failure Cascades Through Service Chain
- **Affected Services**: All / Ingress
- **Fault Type**: Resource
- **Symptom**: A failure in one dependency is followed by rising latency, retries, connection-pool saturation, queue growth, and HTTP 502, 503, or 504 errors across otherwise healthy callers; ingress traffic is stable but internal request volume multiplies; traces show repeated child calls ending at a common dependency
- **Root Cause**: Callers retry or wait on a degraded dependency without bounded deadlines, retry budgets, circuit breaking, back-pressure, or bulkheads. Amplified traffic consumes threads and connections in successive upstream services, turning a local failure into system-wide resource exhaustion
- **Diagnostic Steps**:
  1. Compare ingress request rate with per-service outbound request and retry rates; graph connection-pool utilization, executor queues, timeout counts, and `rate(http_server_requests_seconds_count{status=~"5.."}[5m])` to locate where amplification begins
  2. Use distributed traces to find the earliest failing or slow span shared by affected requests, distinguish the root-cause dependency from victim services, and verify repeated attempts under the same request or idempotency key
  3. Run `kubectl top pods -n <namespace> --containers` and inspect logs for `timeout`, `pool exhausted`, `rejected`, and `circuit` across the suspected dependency and its immediate callers; check EndpointSlice readiness before restarting anything
- **Remediation & Mitigation**: Immediate — stop retry amplification, open circuit breakers, shed optional work, and isolate or roll back the verified root-cause dependency while preserving healthy victims. Long-term — enforce end-to-end deadline and retry budgets, use jitter and idempotency, isolate pools per dependency, add back-pressure and fallbacks, maintain dependency maps, and run controlled cascade-failure tests
