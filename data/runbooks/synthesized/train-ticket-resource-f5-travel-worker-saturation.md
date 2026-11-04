# Train Ticket Resource Faults

## TT-RES-F5-02: Travel Service Worker Saturation from Fan-Out
- **Affected Services**: ts-travel-service, ts-basic-service, ts-route-service, ts-seat-service
- **Fault Type**: Resource
- **Symptom**: Route searches become progressively slower as concurrency rises; `executor_queued_tasks{service="ts-travel-service"}` and `http_server_requests_seconds_bucket` increase before CPU saturation; callers receive HTTP 503 or 504 responses; traces contain parallel route, seat, and basic-info calls followed by a long join delay
- **Root Cause**: A single travel query fans out to several services and waits synchronously for every result. Under load, blocked fan-out tasks consume all request workers and connection slots in `ts-travel-service`, causing queue growth and head-of-line blocking even when individual dependencies are healthy
- **Diagnostic Steps**:
  1. Graph `executor_active_threads{service="ts-travel-service"}`, `executor_queued_tasks{service="ts-travel-service"}`, and `http_client_connections_pending{service="ts-travel-service"}`; compare queue growth with incoming request concurrency rather than CPU alone
  2. Inspect traces for repeated `ts-travel-service` fan-out spans and identify which child span determines the join duration; search logs for pool-acquisition timeouts using the same trace IDs
  3. On one affected replica, run `kubectl exec -n <namespace> <ts-travel-pod> -- kill -3 1` to emit a JVM thread dump, then inspect it with `kubectl logs -n <namespace> <ts-travel-pod> --since=10m | grep -E 'Future.get|WAITING|BLOCKED|pool|timeout'`; confirm replica readiness with `kubectl get pods -n <namespace> -l app=ts-travel-service`
- **Remediation & Mitigation**: Immediate — cap concurrent fan-out, reject excess work with a retryable response, and disable optional child calls while preserving the core route-search path; increase replicas only if dependencies and connection pools have verified headroom. Long-term — apply bulkheads per dependency, bounded queues, partial-result handling, cancellation propagation, and concurrency load tests that validate executor and client-pool sizing together
