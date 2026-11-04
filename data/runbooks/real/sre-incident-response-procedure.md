# Google SRE-Style Incident Response and Emergency Triage Procedure

## SRE-PROC-01: Detection and Alert Triage
- **Affected Services**: All / Ingress
- **Fault Type**: Procedure
- **Symptom**: Customer-facing service-level indicators degrade; HTTP 5xx error ratio exceeds 0.1% or the service-specific SLO threshold; p95 or p99 latency breaches its objective; Prometheus Alertmanager reports active fast-burn or slow-burn alerts
- **Root Cause**: An active but unclassified incident is consuming error budget, and the originating component, scope, and customer impact have not yet been established
- **Diagnostic Steps**:
  1. Open the relevant Prometheus and Alertmanager views and record the alert start time, affected SLO, current error-budget burn rate, and firing labels: `kubectl -n monitoring port-forward svc/alertmanager 9093:9093` and `kubectl -n monitoring port-forward svc/prometheus 9090:9090`
  2. Measure the 5xx ratio by service for the last 5 and 30 minutes using `sum by (service) (rate(http_server_requests_total{status=~"5.."}[5m])) / sum by (service) (rate(http_server_requests_total[5m]))`; repeat with `[30m]` to distinguish a short spike from sustained degradation
  3. Inspect request throughput and latency with `sum by (service) (rate(http_server_requests_total[5m]))` and `histogram_quantile(0.99, sum by (le, service) (rate(http_server_request_duration_seconds_bucket[5m])))`; identify services whose errors or latency changed at the incident onset
  4. Calculate burn rate as `observed_error_ratio / (1 - SLO_target)` and evaluate the configured multi-window alerts; treat 14.4x and 6x as escalation thresholds only when they match the project's documented fast-burn and slow-burn policy
  5. Assign severity from verified impact: P1 for a complete or safety-critical customer outage, P2 for major degraded functionality, or P3 for limited/internal degradation; page the incident commander and service owner according to the escalation policy
- **Remediation & Mitigation**: Immediate — acknowledge the alert, declare the incident, record a timeline, freeze unrelated deployments, and begin component isolation without making an unverified production change. Long-term — maintain tested SLO alerts, severity definitions, ownership metadata, paging routes, and incident-command exercises so detection consistently produces an accountable response

## SRE-PROC-02: Verification and Component Isolation
- **Affected Services**: All / Ingress
- **Fault Type**: Procedure
- **Symptom**: Errors cascade across caller-callee paths; ingress reports HTTP 502, 503, or 504 responses; latency, error, traffic, and saturation signals diverge between upstream services and dependencies
- **Root Cause**: Victim services are propagating dependency failures, making the first failing component difficult to distinguish from downstream or upstream symptoms
- **Diagnostic Steps**:
  1. Separate edge from core failure by comparing ingress errors with client-side dependency errors: query `sum by (service, target, status) (rate(http_client_requests_total{status=~"5.."}[5m]))` and identify the first caller-callee boundary where failures appear
  2. Inspect saturation on candidate services using `rate(container_cpu_usage_seconds_total[5m])`, `rate(container_cpu_cfs_throttled_seconds_total[5m])`, `container_memory_working_set_bytes`, thread-pool queue depth, and database connection-pool utilization; high latency with low local saturation suggests blocking on a dependency or network path
  3. Review recent changes with `kubectl rollout history deployment/<service> -n <namespace>` and `kubectl get events -n <namespace> --sort-by=.lastTimestamp`; correlate rollout revisions, pod restarts, configuration changes, and node events with the incident start time
  4. Follow Jaeger traces from ingress toward dependencies and locate the earliest span that reports an internal error or abnormal self-time; treat later HTTP 504 spans as victims unless their own saturation or error logs show otherwise
  5. Confirm the hypothesis with targeted logs and health checks: `kubectl logs deployment/<service> -n <namespace> --since=30m | grep -iE "error|exception|timeout|panic"` and `kubectl get pods -n <namespace> -l app=<service> -o wide`
- **Remediation & Mitigation**: Immediate — identify the smallest verified failure domain, name the suspected root-cause service and evidence, route or shed traffic away from unhealthy instances when supported, and avoid restarting healthy victim services. Long-term — maintain dependency maps, deployment annotations, trace propagation, saturation dashboards, and per-service ownership so future incidents can be isolated from telemetry rather than intuition

## SRE-PROC-03: Immediate Mitigation and Bleed-Stopping
- **Affected Services**: All / Ingress
- **Fault Type**: Procedure
- **Symptom**: Verified root-cause components continue producing customer-visible errors, timeouts, or saturation, and availability must be restored before full code-level diagnosis is complete
- **Root Cause**: A confirmed release, unhealthy workload, exhausted resource, or failing dependency is actively sustaining the incident
- **Diagnostic Steps**:
  1. Capture a pre-mitigation baseline for 5xx ratio, p99 latency, request rate, ready replicas, pod restarts, and dependency saturation; record the proposed action, owner, expected effect, rollback condition, and observation window in the incident timeline
  2. For a rollout-correlated regression, verify the prior revision with `kubectl rollout history deployment/<service> -n <namespace>`, execute `kubectl rollout undo deployment/<service> -n <namespace>`, and monitor `kubectl rollout status deployment/<service> -n <namespace> --timeout=5m`
  3. For a single deadlocked or memory-exhausted pod, confirm that healthy replicas and disruption safeguards can carry traffic, preserve diagnostics when feasible, and execute `kubectl delete pod <pod-name> -n <namespace>`; verify replacement readiness with `kubectl get pods -n <namespace> -l app=<service> -w`
  4. For capacity saturation, confirm that the database, message broker, and downstream connection budgets can support additional load before executing `kubectl scale deployment/<service> --replicas=<target-count> -n <namespace>`; do not scale a deterministic crash loop or runaway code path
  5. For a cascading dependency failure, activate an approved circuit-breaker, traffic-shift, load-shed, or feature-disable policy; verify the applied service-mesh configuration with `kubectl get destinationrule,virtualservice -n <namespace>` and retain an explicit rollback path
  6. Apply one change at a time and observe at least two complete telemetry scrape/evaluation intervals; continue only if the targeted SLI improves without creating new saturation, data-integrity, or dependency errors
- **Remediation & Mitigation**: Immediate — apply the smallest reversible action that restores the customer-facing SLI, such as rollback, removal of one unhealthy pod, controlled scaling, traffic shedding, or an approved circuit breaker; stop and revert the action if its predefined guardrail is breached. Long-term — replace temporary mitigations with tested code or configuration fixes, encode safe rollback and circuit-breaker controls, and add automation only after the failure mode and guardrails are understood

## SRE-PROC-04: Recovery Verification and Post-Incident RCA
- **Affected Services**: All / Ingress
- **Fault Type**: Procedure
- **Symptom**: Primary customer-facing indicators improve after mitigation, but residual queues, exhausted connection pools, delayed jobs, or recurring failures may still threaten recovery
- **Root Cause**: The initiating failure may be contained while secondary saturation, corrupted state, backlog processing, or an incomplete temporary fix remains
- **Diagnostic Steps**:
  1. Verify that the HTTP 5xx ratio remains below 0.1% or the service-specific SLO threshold for at least 10 consecutive minutes and that p95/p99 latency, traffic, and saturation have returned to their normal ranges
  2. Check ready replicas, restarts, and recent events with `kubectl get deployment,pods -n <namespace> -l app=<service>` and `kubectl get events -n <namespace> --sort-by=.lastTimestamp`; confirm that no crash loop, readiness failure, or eviction continues
  3. Inspect secondary bottlenecks and backlog metrics, including `hikaricp_connections_active`, `hikaricp_connections_pending`, thread-pool queue depth, message-consumer lag, retry volume, and dead-letter queue depth; verify that recovery traffic is draining rather than growing
  4. Export representative Jaeger traces from before, during, and after mitigation; compare the failing span, critical-path duration, status, and dependency calls using the same transaction or trace attributes
  5. Collect bounded logs with `kubectl logs deployment/<service> -n <namespace> --since-time=<incident-start-rfc3339> --timestamps | grep -iE "error|exception|timeout|panic"`; use the centralized log platform to enforce both incident start and end timestamps because `kubectl logs` does not provide an `--until-time` option
  6. Build the post-incident timeline and use the Five Whys to identify the triggering condition, technical root cause, detection gap, response gap, and contributing organizational factors; assign every corrective action an owner, priority, and due date
- **Remediation & Mitigation**: Immediate — keep the incident open until SLIs, saturation, queues, and downstream dependencies remain stable through the observation window; revert temporary capacity or routing changes gradually and monitor each step. Long-term — implement and validate the permanent fix, add regression and failure-injection tests, update alerts and runbooks, track postmortem actions to completion, and conduct a blameless review focused on system improvements
