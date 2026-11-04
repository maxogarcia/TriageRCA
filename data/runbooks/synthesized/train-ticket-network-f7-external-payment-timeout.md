# Train Ticket Network Faults

## TT-NET-F7-01: External Payment Gateway Timeout
- **Affected Services**: ts-inside-payment-service, ts-payment-service
- **Fault Type**: Network
- **Symptom**: Payments that require the external gateway remain pending or fail with HTTP 504, `SocketTimeoutException`, or `PaymentTimeoutException`; `payment_external_timeout_total` rises while internal-balance payments continue to succeed; traces end at the outbound payment span with no timely response
- **Root Cause**: The F7 fallback call from `ts-inside-payment-service` to the external payment endpoint exceeds its connect or response deadline because of network delay, packet loss, gateway overload, or an unreachable upstream. Unbounded or unsafe retries can extend user latency and risk duplicate charges
- **Diagnostic Steps**:
  1. Graph `rate(payment_external_timeout_total[5m])`, `histogram_quantile(0.99, sum by (le) (rate(http_client_request_duration_seconds_bucket{service="ts-inside-payment-service",target="external-payment"}[5m])))`, and outbound connection-error counts
  2. Inspect payment traces for the external span's connect, TLS, and response phases; correlate the payment ID with timeout logs and verify whether multiple attempts reused the same idempotency key
  3. From an authorized ephemeral network-debug container sharing the `ts-inside-payment-service` pod network namespace, run `kubectl debug -n <namespace> -it <inside-payment-pod> --target=<inside-payment-container> --image=<approved-network-debug-image> -- curl -sS -o /dev/null -w '%{http_code} %{time_connect} %{time_total}\n' --max-time <seconds> <external-payment-health-url>`; check DNS and egress policy without submitting a real charge
- **Remediation & Mitigation**: Immediate — open the circuit breaker, place affected payments in a durable pending state, and permit only bounded retries with the original idempotency key; fail over only to an approved gateway. Long-term — use asynchronous payment confirmation and reconciliation, define connect and response budgets, monitor gateway SLOs, test loss and delay behavior, and enforce duplicate-charge protection
