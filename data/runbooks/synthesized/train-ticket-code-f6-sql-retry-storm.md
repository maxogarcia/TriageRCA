# Train Ticket Code-Level Faults

## TT-CODE-F6-01: Voucher SQL Error Triggers Recursive Retry Storm
- **Affected Services**: ts-voucher-service, ts-order-service, ts-order-other-service, ts-ui-dashboard
- **Fault Type**: Code-level
- **Symptom**: Voucher operations first return SQL errors and then degrade into HTTP 504 failures; request rate to `ts-voucher-service`, retry count, database connection usage, and executor queue depth rise without a matching increase in user traffic; traces contain repeated nested calls for the same order
- **Root Cause**: The F6 voucher path executes an invalid SQL statement or argument. The voucher-request caller retries this nonrecoverable failure without a strict attempt budget, multiplying one user request into repeated voucher calls and exhausting the voucher service or its MySQL connection pool; `ts-voucher-service` may call `ts-order-service` or `ts-order-other-service` to retrieve the referenced order, not the reverse
- **Diagnostic Steps**:
  1. Compare `rate(sql_error_total{service="ts-voucher-service"}[5m])`, `rate(retry_attempts_total{target="ts-voucher-service"}[5m])`, `rate(http_server_requests_seconds_count{service="ts-voucher-service"}[5m])`, and the voucher MySQL pool's active and pending connection metrics
  2. Inspect one voucher trace by `order_id`: confirm repeated caller-to-voucher attempts, then identify the first voucher SQL error before secondary timeouts or pool-exhaustion errors; verify whether the voucher lookup called `ts-order-service` or `ts-order-other-service`
  3. Run `kubectl logs -n <namespace> deploy/ts-voucher-service --since=10m | grep -E 'SQL|OperationalError|ProgrammingError|timeout'` and inspect the identified caller's logs for the same `order_id` with `kubectl logs -n <namespace> deploy/<caller-service> --since=10m | grep -E '<order-id>|retry|voucher'`
- **Remediation & Mitigation**: Immediate — stop retries at the identified caller with a circuit breaker or feature flag, roll back or disable the faulty voucher query, and let the backlog and MySQL pool recover before restoring traffic. Long-term — classify SQL syntax and schema errors as nonretryable, cap attempts with jittered backoff and a retry budget, add voucher-query integration tests against the deployed schema, and alert when retry amplification exceeds original request volume
