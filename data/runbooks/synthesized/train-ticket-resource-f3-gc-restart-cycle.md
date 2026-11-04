# Train Ticket Resource Faults

## TT-RES-F3-02: JVM Memory Misconfiguration Causes GC Thrash and Restart Cycle
- **Affected Services**: ts-travel-service, ts-station-service
- **Fault Type**: Resource
- **Symptom**: Travel and station APIs show sawtooth latency, frequent full garbage collections, failed readiness probes, and periodic restarts; `rate(jvm_gc_pause_seconds_sum[5m])` and `jvm_memory_used_bytes{area="heap"}` rise together; upstream services observe HTTP 502 or 503 responses even before an OOM kill occurs
- **Root Cause**: The F3 configuration leaves too little usable heap for the live data set or reserves nearly all container memory for heap. The JVM spends increasing time in full GC, starves health-check and request threads, and is eventually restarted by a failed probe or killed when native overhead exceeds the cgroup limit
- **Diagnostic Steps**:
  1. Graph `rate(jvm_gc_pause_seconds_sum{service=~"ts-travel-service|ts-station-service"}[5m])`, old-generation utilization, readiness, and restart counters; compare total GC pause time with request latency and memory limit headroom
  2. Inspect GC logs for repeated full collections with little reclaimed memory and correlate pause windows with readiness failures and trace latency; check whether the problem began with a JVM flag or resource-limit change
  3. Run `kubectl logs -n <namespace> <pod> --previous` and `kubectl describe pod -n <namespace> <pod>` to identify probe failure versus OOM termination; inspect the running JVM arguments and pod limit with `kubectl exec -n <namespace> <pod> -- sh -c 'tr "\000" " " </proc/1/cmdline; echo'` and `kubectl get pod -n <namespace> <pod> -o jsonpath='{.spec.containers[0].resources.limits.memory}{"\n"}'`
- **Remediation & Mitigation**: Immediate — restore the last known-good heap and container settings, relax only an incorrectly aggressive probe while preserving failure detection, and roll pods gradually to avoid simultaneous capacity loss. Long-term — derive heap and non-heap budgets from load tests, set startup and readiness probes around observed GC behavior, alert on GC time ratio plus memory headroom, and block deployments whose JVM flags conflict with pod limits
