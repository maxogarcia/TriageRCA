# Cross-System Kubernetes Faults

## GEN-K8S-CRASH-01: Application Pod Enters CrashLoopBackOff
- **Affected Services**: All / Ingress
- **Fault Type**: Procedure
- **Symptom**: A workload has fewer ready replicas, pod restart count rises, and container status alternates between `Waiting` with `CrashLoopBackOff` and a terminated state; `kube_pod_container_status_waiting_reason{reason="CrashLoopBackOff"}` equals 1; callers may receive HTTP 502, 503, or 504 responses
- **Root Cause**: The container process repeatedly exits because of an application exception, invalid command or argument, missing configuration or secret, permission error, unavailable startup dependency, or faulty startup/liveness probe; kubelet then applies restart backoff
- **Diagnostic Steps**:
  1. Check `kube_pod_container_status_restarts_total`, `kube_pod_container_status_waiting_reason{reason="CrashLoopBackOff"}`, and Deployment unavailable replicas; correlate the first restart with rollout and configuration annotations
  2. Inspect current and previous container logs plus pod events; identify the first fatal error, exit code, failed probe, or missing dependency rather than treating the backoff itself as the root cause
  3. Run `kubectl describe pod -n <namespace> <pod>`, `kubectl logs -n <namespace> <pod> -c <container> --previous`, and `kubectl get events -n <namespace> --field-selector involvedObject.name=<pod> --sort-by=.lastTimestamp`
- **Remediation & Mitigation**: Immediate — roll back the correlated release or correct the verified command, configuration, secret, permission, dependency, or probe error, then restore traffic only after readiness succeeds. Long-term — validate startup configuration in CI, use tested startup/readiness/liveness probes, retain termination messages, run production-image smoke tests, and alert on restart-rate increases before capacity is lost
