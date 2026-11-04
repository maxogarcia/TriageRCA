# External Dependency Faults

## F6: Recursive Retry Storm from Dependent Service SQL Error
- **Affected Services**: Voucher-Service, Order-Service
- **Fault Type**: Resource
- **Symptom**: Voucher-Service response time degrades progressively until timeout; cascading `504 Gateway Timeout` errors; Prometheus `http_requests_total{status="504",service="voucher-service"}` spikes; upstream caller eventually receives timeout error
- **Root Cause**: A SQL error in Voucher-Service causes every request to fail. The calling service (Order-Service) has a retry-on-failure policy without a circuit breaker. Each failed request triggers an automatic retry, creating an endless recursive loop of requests that saturates Voucher-Service and eventually exhausts connection pools, producing a cascading timeout
- **Diagnostic Steps**:
  1. Check `http_requests_total{service="voucher-service"}` for abnormal request rate spikes; monitor `sql_error_total{service="voucher-service"}` and `retry_attempts_total{service="order-service",target="voucher-service"}` in Prometheus
  2. Search Voucher-Service logs for the originating `SQLException` or `DataAccessException`; in Jaeger, look for deeply nested retry spans from Order-Service → Voucher-Service with exponentially growing trace depth
  3. `kubectl logs -l app=voucher-service --tail=500 | grep -i "sql\|exception"` and `kubectl logs -l app=order-service --tail=500 | grep -c "retry.*voucher"` to quantify retry volume
- **Remediation & Mitigation**: Immediate — fix the SQL error in Voucher-Service; add a circuit breaker (e.g., Resilience4j) in Order-Service with a max-retry limit (3 attempts) and exponential backoff. Long-term — implement a dead-letter queue for failed requests; add health-check probes that detect persistent SQL failures and mark the pod as unready; configure retry budgets across the service mesh

## F7: Third-Party Payment Service Timeout Causing Payment Failure
- **Affected Services**: Inside-Payment-Service, External-Payment-Service
- **Fault Type**: Network
- **Symptom**: Payment fails with timeout error when user balance is insufficient and the system falls back to the third-party payment gateway; HTTP `504` or application-level `PaymentTimeoutException`; Prometheus `payment_external_timeout_total` increments
- **Root Cause**: When a user's internal balance is insufficient, Inside-Payment-Service calls an external third-party Node.js payment service to process the transaction. Under load or network instability, the third-party service experiences delays that exceed the caller's timeout threshold, causing the payment to fail entirely with no fallback
- **Diagnostic Steps**:
  1. Check `payment_external_timeout_total` and `http_request_duration_seconds{service="external-payment-service",quantile="0.99"}` in Prometheus; monitor `payment_fallback_triggered_total` for insufficient-balance scenarios
  2. Inspect Jaeger traces for spans between Inside-Payment-Service and External-Payment-Service; look for spans exceeding the configured timeout (typically 5-10 seconds); search logs for `ConnectTimeoutException` or `SocketTimeoutException`
  3. `kubectl logs -l app=external-payment-service --tail=300 | grep -E "timeout|delay|error"` and `kubectl exec <inside-payment-pod> -- curl -w "@/dev/stdin" -o /dev/null -s http://external-payment-svc:8080/health <<< "\n%{time_total}\n"` to measure round-trip latency
- **Remediation & Mitigation**: Immediate — open the circuit breaker when the external gateway breaches its latency/error threshold, place the payment in a pending state, and show the user a retry-later message; retry only idempotent requests carrying a stable payment idempotency key, with exponential backoff and a maximum of two retries. Long-term — implement an asynchronous payment flow with a durable payment queue, webhook reconciliation, and duplicate-charge protection; define timeout and retry budgets for the external gateway

## F8: Redis Key/Token Not Passed for VIP User Ticket Booking
- **Affected Services**: Auth-Service, Order-Service, Travel-Service
- **Fault Type**: Code-level
- **Symptom**: VIP user ticket booking fails with `401 Unauthorized` or missing-token error at a downstream service; normal user bookings succeed; Prometheus `auth_token_missing_total{user_type="vip"}` increments
- **Root Cause**: VIP users require an additional authentication key/token stored in Redis that must be passed along the booking request chain. A code-level bug causes the VIP-specific token to be dropped when the request is forwarded from the first service to downstream services. The Redis read succeeds, but the token is not included in the outgoing request headers/body to the dependent microservice
- **Diagnostic Steps**:
  1. Check `auth_token_missing_total{user_type="vip"}` and `redis_command_duration_seconds{command="GET",key_prefix="vip_token"}` in Prometheus; compare VIP vs. normal user booking success rates
  2. Inspect Jaeger traces for VIP booking flows; compare request headers between hops — the VIP token should appear in each span's metadata; search service logs for `missing_vip_token` or `unauthorized` entries correlated with VIP user IDs
  3. `kubectl exec <auth-pod> -- redis-cli GET "vip_token:<user_id>"` to verify the token exists in Redis; `kubectl logs -l app=order-service --tail=300 | grep -E "vip|token|unauthorized"` to confirm the token is absent in downstream calls
- **Remediation & Mitigation**: Immediate — fix the token-forwarding logic to include the VIP token in the request propagation chain (e.g., add it to the HTTP header or context object passed between services). Long-term — use a centralized API gateway or service mesh sidecar to inject and propagate authentication tokens automatically; add integration tests covering VIP-specific booking flows end-to-end; implement token-presence assertions in the booking service's request validation layer
