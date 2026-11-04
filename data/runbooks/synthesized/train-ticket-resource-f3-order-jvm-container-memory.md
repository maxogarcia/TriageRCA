# Train Ticket Resource Faults

## TT-RES-F3-01: Order Service JVM Heap Exceeds Container Memory
- **Affected Services**: ts-order-service
- **Fault Type**: Resource
- **Symptom**: `ts-order-service` restarts under load with exit code 137 or reason `OOMKilled`; `container_memory_working_set_bytes` approaches the pod limit while `jvm_memory_used_bytes{area="heap"}` remains below the configured `-Xmx`; order APIs intermittently return HTTP 503 and `kube_pod_container_status_restarts_total` increases
- **Root Cause**: The F3 JVM configuration sets maximum heap too close to or above the Kubernetes memory limit. Heap, metaspace, direct buffers, native libraries, and thread stacks together cross the cgroup limit, so the kernel kills the process before the JVM can report a normal heap exhaustion
- **Diagnostic Steps**:
  1. Compare `container_memory_working_set_bytes{pod=~"ts-order-service.*"}` with `kube_pod_container_resource_limits{resource="memory",pod=~"ts-order-service.*"}` and compare JVM heap usage with `jvm_memory_max_bytes`; check `kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}`
  2. Correlate order-service restart timestamps with GC logs and failed order traces; distinguish a cgroup kill with no Java exception from `java.lang.OutOfMemoryError` recorded before termination
  3. Run `kubectl describe pod -n <namespace> <pod>` to verify `OOMKilled`, inspect the actual PID 1 JVM arguments with `kubectl exec -n <namespace> <pod> -- sh -c 'tr "\000" " " </proc/1/cmdline; echo'`, and compare them with `kubectl get pod -n <namespace> <pod> -o jsonpath='{.spec.containers[0].resources.limits.memory}{"\n"}'`
- **Remediation & Mitigation**: Immediate — roll back the bad JVM settings or set heap to a measured fraction of the container limit, restart one replica at a time, and keep traffic on healthy pods. Long-term — use container-aware JVM sizing such as `-XX:MaxRAMPercentage`, budget native and stack memory explicitly, validate heap-to-limit ratios in deployment policy, and load-test peak order volume before release
