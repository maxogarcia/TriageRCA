# Train Ticket Code-Level Faults

## TT-CODE-F18-01: Null Food Response Crashes Frontend Rendering
- **Affected Services**: ts-food-service, ts-train-food-service, ts-ui-dashboard
- **Fault Type**: Code-level
- **Symptom**: The food-selection view fails only for trains with no configured train food; the API returns HTTP 200 but frontend error telemetry reports a null dereference such as `TypeError`; traces show a response field containing `null` where the UI expects a collection
- **Root Cause**: The F18 food lookup path returns an absent train-food result as `null`, and the frontend immediately indexes or iterates it without a null guard. The producer and consumer therefore disagree on the empty-result contract even though transport and HTTP status are successful
- **Diagnostic Steps**:
  1. Compare frontend exception counts with requests for trains that have no food records; if custom instrumentation exports `food_query_null_response_total`, graph it as supporting evidence, otherwise count the equivalent structured null-response log event; verify that service availability and HTTP error metrics remain normal
  2. Inspect a failed trace's sanitized response shape and browser error context; compare it with a successful train and confirm that only the empty collection representation differs
  3. From an approved debug pod, run `curl -sS http://ts-food-service:18856/api/v1/foodservice/foods/<date>/<start-station>/<end-station>/<trip-id-without-food>` and verify whether the train-food field is `null` rather than `[]`; correlate the request with `ts-food-service` and `ts-train-food-service` logs
- **Remediation & Mitigation**: Immediate — add a frontend null guard and return an empty collection from `ts-food-service`; roll out producer and consumer changes compatibly and verify a train with no food. Long-term — publish a non-null API schema, validate responses, generate typed clients, add empty-state UI tests and consumer contracts, and monitor schema violations independently of HTTP status
