# Configuration Faults

## F3: JVM Memory Limit Exceeds Docker Container Memory
- **Affected Services**: Order-Service, Travel-Service, Station-Service
- **Fault Type**: Config
- **Symptom**: Microservices become periodically unavailable; HTTP 500 or 503 errors spike; Prometheus `up{job="<service>"}` flaps between 0 and 1; container restarts visible in `kube_pod_container_status_restarts_total`
- **Root Cause**: JVM `-Xmx` heap setting is configured larger than the Docker container's memory limit. Under load the JVM attempts to allocate beyond the cgroup limit, causing the Linux OOM killer to terminate the JVM process. Kubernetes then restarts the pod, creating a crash-loop cycle
- **Diagnostic Steps**:
  1. Check `kube_pod_container_status_restarts_total` and `container_memory_usage_bytes` vs. `container_spec_memory_limit_bytes` in Prometheus for affected pods
  2. Search pod logs for `OOMKilled` events; run `kubectl describe pod <pod>` and look for `Last State: Terminated, Reason: OOMKilled`
  3. `kubectl get pod <pod> -o jsonpath='{.spec.containers[0].resources.limits.memory}'` and compare with JVM flags: `kubectl exec <pod> -- jcmd 1 VM.flags | grep -i "MaxHeapSize"`
- **Remediation & Mitigation**: Immediate — set JVM `-Xmx` to no more than 75% of the Docker/K8s memory limit to leave room for metaspace, thread stacks, and OS overhead. Long-term — use JVM flags `-XX:+UseContainerSupport` and `-XX:MaxRAMPercentage=75.0` so the JVM auto-detects container limits; add memory-limit validation to the CI/CD pipeline

## F4: SSL Offloading Applied Per-Microservice Causing Latency
- **Affected Services**: All / Ingress
- **Fault Type**: Config
- **Symptom**: Response times for multi-hop requests are 3-10x longer than expected; Prometheus `http_request_duration_seconds{quantile="0.99"}` is elevated across all services; no error codes, just slow responses
- **Root Cause**: SSL/TLS termination is configured on every individual microservice rather than at the ingress gateway. Each inter-service hop performs a full TLS handshake, multiplying cryptographic overhead across the call chain. Complex flows that traverse 5+ services accumulate significant latency
- **Diagnostic Steps**:
  1. Compare `http_request_duration_seconds` across services in Prometheus; check if latency scales linearly with call-chain depth
  2. Inspect Jaeger traces for TLS-handshake spans between services; look for `ssl_handshake_duration` or connection-setup time dominating span durations
  3. `kubectl exec <pod> -- openssl s_client -connect <downstream-service>:443 -brief 2>&1 | head -5` to confirm TLS is active on internal service ports; `kubectl get svc -o wide` to check port configurations
- **Remediation & Mitigation**: Immediate — move TLS termination to the ingress controller (e.g., Nginx Ingress or Istio Gateway) and use plaintext or mTLS with sidecar proxies for internal traffic. Long-term — deploy a service mesh (Istio/Linkerd) that handles mTLS transparently at the sidecar level, eliminating per-service SSL configuration overhead

## F15: Nginx Request Body Size Limit Blocks Composite Requests
- **Affected Services**: Nginx-Gateway, Food-Service, Consign-Service
- **Fault Type**: Config
- **Symptom**: HTTP `413 Request Entity Too Large` when users submit orders with food selection and consignment together; Prometheus `nginx_http_requests_total{status="413"}` increases; smaller requests succeed normally
- **Root Cause**: The `client_max_body_size` in `nginx.conf` is set to 200 bytes. When a user selects food items and adds consignment details, the combined JSON request body exceeds this limit, and Nginx rejects the request before it reaches the backend service
- **Diagnostic Steps**:
  1. Check `nginx_http_requests_total{status="413"}` in Prometheus; correlate with endpoints involving food + consignment
  2. Search Nginx access logs for `413` status codes: identify which URIs are affected and compare request `Content-Length` headers against the configured limit
  3. `kubectl exec <nginx-pod> -- cat /etc/nginx/nginx.conf | grep -i "client_max_body_size"` and `kubectl logs -l app=nginx-gateway --tail=200 | grep "413"`
- **Remediation & Mitigation**: Immediate — increase `client_max_body_size` to a reasonable value (e.g., `10m`) in `nginx.conf` and reload Nginx. Long-term — externalize Nginx configuration into a ConfigMap with validated defaults; add integration tests that submit maximum-payload requests to catch regressions

## F16: File Upload Rejected Due to Max Content Length Limit
- **Affected Services**: Route-Admin-Service, Nginx-Gateway
- **Fault Type**: Config
- **Symptom**: Batch route upload via file fails with `413 Payload Too Large` or application-level rejection; small files upload successfully; Prometheus `upload_failures_total{reason="size_exceeded"}` increments
- **Root Cause**: The `max-content-length` configuration (originally from Spray framework, replicated in the TrainTicket gateway) is set to only 2 MB. Route batch-upload files that exceed this threshold are rejected at the gateway or framework level before reaching the application handler
- **Diagnostic Steps**:
  1. Check `upload_failures_total{reason="size_exceeded"}` and `http_requests_total{status="413",path="/api/v1/routeservice/routes/upload"}` in Prometheus
  2. Search Route-Admin-Service and gateway logs for `EntityTooLarge` or `max-content-length exceeded` error messages
  3. `kubectl exec <route-admin-pod> -- cat /app/application.yml | grep -i "max-content-length"` and test with: `curl -v -F "file=@largefile.csv" http://<route-admin-svc>/api/v1/routeservice/routes/upload`
- **Remediation & Mitigation**: Immediate — increase `max-content-length` to accommodate expected batch file sizes (e.g., 50 MB) in both the application config and any upstream proxy. Long-term — implement chunked/streaming upload support for large batch files; add client-side file-size validation with a clear error message before submission

## F20: Shared Library Version Mismatch Across Microservices
- **Affected Services**: Order-Service, Travel-Service, Voucher-Service
- **Fault Type**: Config
- **Symptom**: Workflow data requests return empty results or wrong data; order status values are misinterpreted between services; Prometheus `order_status_parse_error_total` or `grpc_status{code="INTERNAL"}` increments
- **Root Cause**: Multiple microservices depend on a shared library (`ts-common`) that defines core data structures such as the Order status enum. Different services are compiled against different versions of this library, where the same enum constant maps to different integer values. When Service A writes a status using v2 enum values and Service B reads it using v1 enum mapping, the status is misinterpreted
- **Diagnostic Steps**:
  1. Check `order_status_parse_error_total` or unexpected `null`/empty responses in Prometheus; compare error rates across services that share the library
  2. Search service logs for deserialization errors, `EnumConstantNotPresentException`, or unexpected status values; diff trace payloads between producer and consumer services
  3. `kubectl exec <pod> -- ls /app/lib/ | grep ts-common` across multiple service pods to compare library versions; `kubectl exec <pod> -- unzip -p /app/lib/ts-common-*.jar META-INF/MANIFEST.MF | grep Version`
- **Remediation & Mitigation**: Immediate — align all services to the latest `ts-common` library version and redeploy. Long-term — enforce shared-library version pinning in the build system (Maven BOM or Gradle platform); add CI checks that fail the build if any service uses a mismatched version; consider using a schema registry for cross-service data contracts
