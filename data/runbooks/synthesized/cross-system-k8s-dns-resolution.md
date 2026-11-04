# Cross-System Kubernetes Network Faults

## GEN-K8S-DNS-01: Service DNS Resolution Failure
- **Affected Services**: All / Ingress
- **Fault Type**: Network
- **Symptom**: Pods report `NXDOMAIN`, `SERVFAIL`, `no such host`, or DNS timeouts before connecting to dependencies; direct ClusterIP access may work; `coredns_dns_responses_total{rcode=~"SERVFAIL|REFUSED"}` or DNS latency rises and application timeout metrics increase across multiple services
- **Root Cause**: CoreDNS pods or the `kube-dns` Service are unhealthy, the Corefile or upstream forwarder is invalid, EndpointSlices are missing, pod `dnsPolicy` or search domains are wrong, or NetworkPolicy blocks UDP/TCP port 53
- **Diagnostic Steps**:
  1. Graph CoreDNS request latency and response codes, CoreDNS pod readiness/restarts, and affected application name-resolution errors; determine whether failures are cluster-wide, namespace-specific, or limited to external names
  2. Correlate application logs with CoreDNS logs for the queried name and response code; compare service-name resolution with direct ClusterIP connectivity to separate DNS from transport failure
  3. From an approved ephemeral network-debug container attached to the affected pod, run `kubectl debug -n <namespace> -it <affected-pod> --target=<container> --image=<approved-network-debug-image> -- cat /etc/resolv.conf` and repeat with `nslookup <service>.<namespace>.svc.cluster.local`; inspect `kubectl get pods -n kube-system -l k8s-app=kube-dns`, `kubectl get svc -n kube-system kube-dns -o wide`, `kubectl get endpointslice -n kube-system -l kubernetes.io/service-name=kube-dns`, and the relevant NetworkPolicies
- **Remediation & Mitigation**: Immediate — restore healthy CoreDNS replicas and EndpointSlices, roll back the faulty Corefile or pod DNS policy, and add only the required DNS policy allowance; remove debug pods after verification. Long-term — version and test DNS configuration, autoscale CoreDNS, monitor error codes and latency, define DNS rules in default-deny templates, and consider NodeLocal DNSCache where query volume justifies it
