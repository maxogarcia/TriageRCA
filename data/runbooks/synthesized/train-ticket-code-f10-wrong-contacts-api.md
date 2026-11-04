# Train Ticket Code-Level Faults

## TT-CODE-F10-01: Reservation Flow Calls the Wrong Contacts API
- **Affected Services**: ts-preserve-service, ts-contacts-service
- **Fault Type**: Code-level
- **Symptom**: A specific contact type or reservation path fails with HTTP 400, 404, 405, or 500 while ordinary reservations succeed; `reservation_failures_total{reason="contact_api_error"}` is concentrated in one request subtype; traces show an unexpected contacts endpoint or response schema
- **Root Cause**: The F10 branch-selection logic in `ts-preserve-service` maps an edge-case reservation to the wrong `ts-contacts-service` operation. The called endpoint expects different parameters or semantics, so its response cannot be used by the reservation workflow
- **Diagnostic Steps**:
  1. Break down preserve failures by contact type and booking path; if custom instrumentation exports `reservation_failures_total{reason="contact_api_error"}`, graph `rate(reservation_failures_total{reason="contact_api_error"}[5m])`, otherwise derive the cohort from structured preserve-service error logs and compare it with total preserve requests
  2. Diff successful and failed traces at the contacts call, including route template, method, sanitized request shape, response status, and deployed versions; correlate the failing branch with preserve-service logs
  3. Run `kubectl logs -n <namespace> deploy/ts-preserve-service --since=15m | grep '<trace-id>'` and `kubectl logs -n <namespace> deploy/ts-contacts-service --since=15m | grep '<trace-id>'` to verify the selected endpoint and rejection reason
- **Remediation & Mitigation**: Immediate — route the affected scenario to the correct contacts operation or disable only that path while preserving standard reservations; roll back if the mapping changed in the current release. Long-term — use a typed client with one explicit discriminator, validate inputs before dispatch, add consumer-driven contract tests for every branch, and include the edge case in end-to-end reservation tests
