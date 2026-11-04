# Train Ticket Network Faults

## TT-NET-F1-01: Cancellation Events Complete Out of Order Under Network Delay
- **Affected Services**: ts-cancel-service, ts-inside-payment-service, ts-order-service
- **Fault Type**: Network
- **Symptom**: Cancelled orders remain in `CANCELLING` or revert from `CANCEL` to an earlier state; `order_status_mismatch_total` rises during latency incidents; traces show the refund path and final status-reset call completing in a different order from the cancellation workflow's intended sequence
- **Root Cause**: The F1 workflow runs refund and order-status work asynchronously without enforcing completion order. Variable network delay lets the final `CANCEL` update complete before the delayed refund path finishes its earlier `CANCELLING` write, allowing the stale transition to overwrite the correct state
- **Diagnostic Steps**:
  1. Compare `order_status_transitions_total{from="CANCELLING",to="CANCEL"}` with cancellation volume and graph HTTP client latency, timeout count, and retry count across `ts-cancel-service`, `ts-inside-payment-service`, and the applicable order service
  2. Reconstruct one affected order from trace and log timestamps using `order_id`, trace ID, operation name, and completion time; verify that the delayed `CANCELLING` write was accepted after the final `CANCEL` transition
  3. Run `kubectl logs -n <namespace> -l app=ts-cancel-service --since=15m | grep '<order-id>'` and repeat for `ts-inside-payment-service`, `ts-order-service`, and `ts-order-other-service`; correlate the entries with the matching trace spans and pod restart timestamps
- **Remediation & Mitigation**: Immediate — pause or disable the faulty cancellation path, reconcile affected orders from the authoritative payment result, and reject stale state writes using a version check. Long-term — encode cancellation as an ordered saga or state machine, make each step idempotent, attach monotonic state versions, and test delay, duplication, timeout, and retry behavior explicitly
