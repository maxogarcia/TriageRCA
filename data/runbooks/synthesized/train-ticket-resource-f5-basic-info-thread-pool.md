# Train Ticket Resource Faults

## TT-RES-F5-01: Basic Info Thread Pool Exhaustion
- **Affected Services**: ts-basic-service, ts-travel-service, ts-ticketinfo-service
- **Fault Type**: Resource
- **Symptom**: Ticket-information requests slow down or return HTTP 504 errors during burst traffic; `jvm_threads_live_threads{service="ts-basic-service"}` and the application worker-pool active count reach their configured maxima; executor queue depth rises while pod CPU can remain below its limit; traces show long gaps before work begins in `ts-basic-service`
- **Root Cause**: The F5 asynchronous ticket-information path submits blocking dependency calls to a bounded shared executor. Slow tasks occupy every worker, new tasks accumulate in the queue or are rejected, and callers time out before the service can schedule their work
- **Diagnostic Steps**:
  1. Compare `executor_active_threads{service="ts-basic-service"}` with `executor_pool_max_threads` and graph `executor_queued_tasks{service="ts-basic-service"}` alongside `rate(http_server_requests_seconds_count{service="ts-basic-service",status=~"5.."}[5m])`
  2. Inspect a failed distributed trace from `ts-travel-service` through `ts-basic-service`; confirm that the server span has high scheduling or queue wait and correlate it with `RejectedExecutionException` or `TimeoutException` in the same trace ID
  3. On one affected replica, run `kubectl exec -n <namespace> <ts-basic-pod> -- kill -3 1` to emit a JVM thread dump to container logs, then inspect it with `kubectl logs -n <namespace> <ts-basic-pod> --since=10m | grep -E 'RUNNABLE|WAITING|BLOCKED|RejectedExecutionException|TimeoutException'`; use an approved JDK diagnostic container only if `SIGQUIT` dumps are disabled
- **Remediation & Mitigation**: Immediate — shed nonessential ticket-info traffic, open a circuit breaker for the slow dependency, and scale only after confirming downstream capacity; if a release changed executor behavior, roll it back. Long-term — isolate blocking dependencies in separate bounded pools, enforce per-call deadlines and queue limits, add back-pressure or fallback responses, size executors from measured concurrency, and load-test the F5 path
