# Train Ticket Network Faults

## TT-NET-F1-02: Delayed Cancellation Call Leaves Partial Workflow
- **Affected Services**: ts-cancel-service, ts-inside-payment-service, ts-order-service
- **Fault Type**: Network
- **Symptom**: A user receives a cancellation acknowledgement but the refund or final order-state update does not complete; `cancellation_step_timeout_total` rises; one service logs success while the next service has no matching server span; repeated user attempts can trigger duplicate refund or status-change calls
- **Root Cause**: A delayed or interrupted F1 cancellation call between `ts-cancel-service`, `ts-inside-payment-service`, and an order service exceeds its deadline, while the workflow treats dispatch or timeout as a definitive outcome. The caller cannot tell whether the remote side committed the step, so an unsafe retry can leave the cancellation partially completed or duplicated
- **Diagnostic Steps**:
  1. Graph cancellation and refund step-completion counters, `rate(cancellation_step_timeout_total[5m])`, and HTTP client latency/error rates for calls from `ts-cancel-service` to `ts-inside-payment-service` and the applicable order service; compare completed steps by `order_id`
  2. Search traces and structured logs for the same `order_id` and trace ID; a client timeout with no server span suggests non-delivery, while a completed server span with a timed-out client span indicates an unknown outcome that must be reconciled before retry
  3. Run `kubectl logs -n <namespace> deploy/ts-cancel-service --since=30m | grep '<order-id>'`, repeat for `ts-inside-payment-service` and the applicable order deployment, and verify endpoints plus recent pod restarts with `kubectl get pods -n <namespace> -l 'app in (ts-cancel-service,ts-inside-payment-service,ts-order-service,ts-order-other-service)' -o wide`
- **Remediation & Mitigation**: Immediate — reconcile the authoritative order and refund records, replay only a confirmed-missing idempotent step, and throttle duplicate user retries until the workflow is consistent. Long-term — persist cancellation progress as a recoverable saga, require idempotency keys and monotonic state versions, use bounded retries for transient failures only, and run an automated reconciler for incomplete cancellations
