# Concurrency Faults

## F1: Asynchronous Message Sequence Loss in Ticket Cancellation
- **Affected Services**: Cancel-Service, Inside-Payment-Service, Order-Service
- **Fault Type**: Concurrency
- **Symptom**: Order status stuck at CANCELLING instead of CANCEL after ticket cancellation; intermittent order-state inconsistency visible in `order_status` database column; Prometheus metric `cancel_request_duration_seconds` shows high p99 latency spikes
- **Root Cause**: The cancellation workflow fires two async events — [Drawback Money] and [Reset Order Status] — without sequence control. Under network congestion the money-drawback event completes after the status-reset event, causing the final status write to be overwritten by the earlier event's stale CANCELLING value
- **Diagnostic Steps**:
  1. Check `cancel_request_duration_seconds{service="cancel-service"}` and `order_status_transitions_total{from="CANCELLING",to="CANCEL"}` in Prometheus for drops in successful transitions
  2. Search Cancel-Service and Inside-Payment-Service logs for interleaved `drawback_complete` and `status_reset` timestamps; inspect Jaeger traces for span ordering anomalies in the cancellation flow
  3. `kubectl logs -l app=cancel-service --tail=500 | grep -E "drawback|status_reset"` and `kubectl logs -l app=inside-payment-service --tail=500 | grep "order_status"`
- **Remediation & Mitigation**: Immediate — add a distributed lock or ordered message queue (e.g., Kafka with partition-key on orderId) to enforce [Drawback Money] completes before [Reset Order Status]. Long-term — refactor to a saga/choreography pattern with explicit state machine transitions and idempotent status updates

## F2: Out-of-Order Report Display from Concurrent Reservation Requests
- **Affected Services**: Reservation-Service, UI-Frontend
- **Fault Type**: Concurrency
- **Symptom**: Search results or reservation confirmations display stale or mismatched ticket data when multiple requests are submitted rapidly; users see another request's result; no HTTP errors but UI content is incorrect
- **Root Cause**: Multiple reservation requests sent in rapid succession return in a different order than dispatched due to per-request random network delay. The frontend renders whichever response arrives last without checking request sequence, overwriting newer results with older ones
- **Diagnostic Steps**:
  1. Monitor `http_request_duration_seconds{service="reservation-service"}` histogram for high variance; check `reservation_requests_total` rate for burst patterns
  2. Correlate frontend network waterfall (browser DevTools or trace spans) with backend Jaeger traces; look for response timestamps arriving out of dispatch order
  3. `kubectl logs -l app=reservation-service --tail=300 | grep "request_id"` to verify request-to-response ordering
- **Remediation & Mitigation**: Immediate — implement request sequence IDs on the frontend; discard responses whose sequence number is older than the latest rendered. Long-term — add server-side request deduplication and use optimistic concurrency control with version tokens on displayed data

## F5: Thread Pool Exhaustion Causing Cross-Service Timeout
- **Affected Services**: Basic-Info-Service, Travel-Service, Order-Service
- **Fault Type**: Concurrency
- **Symptom**: Timeout exceptions (`504 Gateway Timeout` or application-level `TimeoutException`) on ticket-info queries when concurrent load is high; Prometheus `thread_pool_active_threads{service="basic-info-service"}` equals `max_thread_pool_size`
- **Root Cause**: Basic-Info-Service uses a shared thread pool for async ticket-information collection. Under burst traffic the pool saturates — all threads are occupied by long-running requests — causing new incoming requests from other services (e.g., Travel-Service) to queue and eventually time out
- **Diagnostic Steps**:
  1. Check `thread_pool_active_threads` and `thread_pool_queue_size` metrics in Prometheus for Basic-Info-Service; alert when active threads reach `max_thread_pool_size`
  2. Search logs for `RejectedExecutionException` or `TimeoutException` in Basic-Info-Service; inspect Jaeger traces for long queue-wait spans
  3. `kubectl exec -it $(kubectl get pod -l app=basic-info-service -o jsonpath='{.items[0].metadata.name}') -- jstack 1 | grep -A5 "pool-"` to inspect thread states
- **Remediation & Mitigation**: Immediate — increase `max-thread-pool-size` and add a bulkhead (separate thread pools per caller). Long-term — implement circuit breakers (Hystrix/Resilience4j) with fallback responses and back-pressure signaling; consider reactive/non-blocking I/O to eliminate thread-per-request bottlenecks

## F11: Non-Deterministic Sequence in Dual-Service Database Write
- **Affected Services**: Order-Service, Cancel-Service
- **Fault Type**: Concurrency
- **Symptom**: Order status intermittently shows incorrect value after cancellation; issue is non-deterministic — sometimes the value self-corrects, sometimes it persists; Prometheus `order_status_mismatch_total` increments sporadically
- **Root Cause**: Two microservices write the same order-status field in the database during the cancellation flow without coordination. When they execute out of order, the value is wrong. A probabilistic recheck in the second writer sometimes corrects it, but when the recheck does not fire, the stale value persists
- **Diagnostic Steps**:
  1. Query `order_status_mismatch_total` and `order_update_duration_seconds` in Prometheus; correlate spikes with cancellation request volume
  2. Search Order-Service and Cancel-Service logs for concurrent `UPDATE orders SET status=` statements with overlapping timestamps on the same `order_id`; check if `recheck_triggered=true` log entries are absent during fault windows
  3. `kubectl exec -it $(kubectl get pod -l app=order-service -o jsonpath='{.items[0].metadata.name}') -- mysql -e "SELECT order_id, status, updated_at FROM orders WHERE status != 'CANCEL' AND updated_at > NOW() - INTERVAL 10 MINUTE"`
- **Remediation & Mitigation**: Immediate — add optimistic locking (version column) on the orders table so the second write fails if the row was modified. Long-term — consolidate status ownership to a single service using a saga coordinator; remove the probabilistic recheck in favor of deterministic compensation logic

## F12: Status-Dependent Fault in Order Cancellation (Locked Stations)
- **Affected Services**: Order-Service
- **Fault Type**: Concurrency
- **Symptom**: Cancellation requests return `403 Forbidden` or a business-error response for specific station pairs while other cancellations succeed; Prometheus `order_cancel_rejected_total{reason="locked_station"}` spikes during admin operations
- **Root Cause**: When an admin operates on orders between two stations, those stations are added to a `locked_station` list in Order-Service. Concurrent user cancellation requests whose start or terminal station is in the locked list are rejected. The lock is held for the duration of the admin operation and interacts with the async thread pool, creating a compound concurrency issue
- **Diagnostic Steps**:
  1. Check `order_cancel_rejected_total{reason="locked_station"}` and `admin_operation_in_progress` gauge in Prometheus
  2. Search Order-Service logs for `station_locked` rejection entries; correlate timestamps with admin-operation audit logs
  3. `kubectl logs -l app=order-service --tail=500 | grep -E "locked_station|admin_lock"` and `kubectl exec -it <order-pod> -- curl localhost:8080/api/v1/orderservice/locked-stations`
- **Remediation & Mitigation**: Immediate — implement a read-write lock so user cancellations queue rather than reject outright; add a TTL on station locks to prevent indefinite blocking. Long-term — decouple admin batch operations into an async job that processes orders without holding a global lock; use event-driven architecture to notify dependent flows when locks release

## F13: Race Condition Between Payment and Cancellation Flows
- **Affected Services**: Cancel-Service, Inside-Payment-Service, Order-Service, UI-Frontend
- **Fault Type**: Concurrency
- **Symptom**: Cancellation confirmation succeeds but payment was never completed, or payment completes after cancellation, leaving the order in an inconsistent paid-but-cancelled state; Prometheus `order_state_conflict_total` increments
- **Root Cause**: A user rapidly triggers both the booking-payment and cancellation flows. Each flow comprises multiple sequential requests (login, search, select, confirm, pay, cancel) with variable network delays. If the cancellation-confirm request begins processing before the payment request completes, the two flows interleave, causing state corruption
- **Diagnostic Steps**:
  1. Monitor `order_state_conflict_total` and compare `payment_completed_total` vs. `cancel_completed_total` rates in Prometheus for the same order IDs
  2. In Jaeger, search for traces where `payment.confirm` and `cancel.confirm` spans overlap for the same `order_id`; check Order-Service logs for rapid status transitions (e.g., NOTPAID → PAID → CANCEL within seconds)
  3. `kubectl logs -l app=cancel-service --tail=500 | grep "order_id=<suspect_id>"` and `kubectl logs -l app=inside-payment-service --tail=500 | grep "order_id=<suspect_id>"`
- **Remediation & Mitigation**: Immediate — add a mutex/pessimistic lock per `order_id` so payment and cancellation cannot execute concurrently on the same order. Long-term — implement a finite state machine for order lifecycle with guard conditions that reject cancellation while payment is in-flight; use an event-sourcing pattern to maintain a complete audit trail of state transitions
