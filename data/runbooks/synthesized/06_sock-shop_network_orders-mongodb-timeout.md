# Sock Shop Network Faults

## SS-NET-MONGO-001: Orders-to-MongoDB Network Delay Causes Timeouts
- **Affected Services**: orders, orders-db
- **Fault Type**: Network
- **Symptom**: Creating or listing orders becomes slow or fails; orders logs contain Java MongoDB driver selection, connection, or socket timeout exceptions; the verified `http_request_duration_seconds` histogram for orders shifts upward; request threads can remain blocked while operations wait on `orders-db:27017`
- **Root Cause**: Added network delay or intermittent reachability between the Java orders service and MongoDB at `orders-db:27017` pushes operations beyond driver selection or I/O timeouts. Waiting database calls consume request capacity and propagate latency or errors to front-end
- **Diagnostic Steps**:
  1. Query `histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket{service="orders"}[5m])))`; if custom instrumentation exports `mongodb_driver_pool_wait_seconds`, inspect it; otherwise derive the pool-wait signal from structured timeout logs and concurrent request latency
  2. If tracing exports MongoDB client spans, inspect slow traces for database wait; otherwise correlate `MongoTimeoutException`, server-selection, socket-timeout, or connection messages from `kubectl logs deployment/orders -n <namespace> --since=30m` with orders-db logs and authorized slow-query/profiler evidence to distinguish network delay from slow queries
  3. Verify `kubectl get service orders-db -n <namespace>` and `kubectl get endpointslice -n <namespace> -l kubernetes.io/service-name=orders-db`; from an authorized ephemeral debug container attached to an orders pod, resolve `orders-db` and run repeated TCP-connect timing checks to port 27017, then inspect relevant NetworkPolicies and the authorized delay-injection controller
- **Remediation & Mitigation**: Immediate — remove the confirmed delay experiment or policy fault, restore healthy orders-db endpoints, shed nonessential reads, and restart only clients whose MongoDB pools fail to recover; do not mask the incident by blindly extending timeouts. Long-term — set bounded selection and socket timeouts within the end-to-end budget, add pool-wait instrumentation, isolate database waits with bulkheads, use bounded retries only for safe idempotent operations, and test delayed-database behavior under load
