# Data Faults

## F17: Nested SQL Causing Slow Query Timeout in Voucher Service
- **Affected Services**: Voucher-Service, UI-Frontend
- **Fault Type**: Data
- **Symptom**: Voucher query requests time out after 5 seconds on the frontend; HTTP 504 or client-side timeout error displayed; Prometheus `http_request_duration_seconds{service="voucher-service",quantile="0.99"}` exceeds 10 seconds
- **Root Cause**: The SQL query in Voucher-Service contains excessively nested `SELECT` and `FROM` sub-clauses (simulated as a MySQL stored procedure with a 10-second `SLEEP`). The frontend enforces a 5-second network timeout, so the query always times out before the database returns results
- **Diagnostic Steps**:
  1. Check `http_request_duration_seconds{service="voucher-service"}` and `mysql_slow_queries_total` in Prometheus for sustained high-latency queries
  2. Search Voucher-Service logs for `QueryTimeoutException` or `StatementTimeout`; inspect slow-query logs in MySQL for queries exceeding 5 seconds
  3. `kubectl exec -it $(kubectl get pod -l app=voucher-service -o jsonpath='{.items[0].metadata.name}') -- mysql -e "SHOW PROCESSLIST"` and `kubectl exec <pod> -- mysql -e "SHOW PROCEDURE STATUS WHERE Db='voucher_db'"`
- **Remediation & Mitigation**: Immediate — remove or optimize the nested sub-queries; eliminate the artificial `SLEEP` procedure; rewrite as flat JOINs or use indexed views. Long-term — add query execution time limits at the application layer; implement database query review in CI/CD; set `max_execution_time` in MySQL configuration to kill runaway queries

## F18: Null JSON Value Crashes Frontend Chart Rendering
- **Affected Services**: Food-Service, UI-Frontend
- **Fault Type**: Data
- **Symptom**: Product analysis chart fails to render or displays an error when a train has no associated food items; browser console shows `TypeError: Cannot read property of null`; Prometheus `http_requests_total{status="200",path="/api/v1/foodservice/foods"}` returns 200 but the UI breaks
- **Root Cause**: When a train has no food configured, the `getFood` API returns `null` for the train-food key in the JSON response. The frontend attempts to access the first item of this null value without a null-check, causing a JavaScript runtime error that breaks the chart component
- **Diagnostic Steps**:
  1. Check `food_query_null_response_total` or frontend error tracking metrics in Prometheus/Sentry for null-related exceptions on the food endpoint
  2. Inspect Food-Service response payloads in Jaeger traces for trains without food; search browser console logs (or frontend error logs) for `TypeError: Cannot read property` related to food data
  3. `kubectl exec <food-service-pod> -- curl localhost:8080/api/v1/foodservice/foods/<train-id-without-food>` to reproduce the null response; verify the JSON structure
- **Remediation & Mitigation**: Immediate — add null-checks and default empty-array fallbacks in the frontend before accessing food data (`const trainFood = result?.[0] ?? []`). Long-term — enforce non-null API contracts: Food-Service should return an empty array `[]` instead of `null` when no food exists; add JSON schema validation to API responses

## F19: Price Displayed in Wrong Locale Format (French)
- **Affected Services**: Consign-Service, UI-Frontend
- **Fault Type**: Data
- **Symptom**: Product price displays in an unexpected format when French locale is selected (e.g., `1.234,56` instead of `1 234,56 €`); no HTTP errors; visual inconsistency reported by users
- **Root Cause**: The price formatting logic does not correctly apply French locale conventions. The `NumberFormat` or equivalent formatter uses an incorrect locale string or missing locale-specific grouping/decimal separators, producing malformed price strings for `fr-FR` locale
- **Diagnostic Steps**:
  1. Check application-level metrics for locale-specific request volumes: `http_requests_total{locale="fr-FR"}` in Prometheus; no error-rate anomaly expected since this is a formatting bug, not an exception
  2. Inspect Consign-Service response payloads in Jaeger traces when `Accept-Language: fr-FR` is set; compare the `price` field format against expected French number formatting (`1 234,56 €`)
  3. `kubectl exec <consign-pod> -- curl -H "Accept-Language: fr-FR" localhost:8080/api/v1/consignservice/consigns/<id>` and inspect the returned price string
- **Remediation & Mitigation**: Immediate — fix the locale formatter to use `Locale.FRANCE` (Java) or `Intl.NumberFormat('fr-FR', {style: 'currency', currency: 'EUR'})` (JavaScript) for French price formatting. Long-term — centralize number/currency formatting in a shared utility with unit tests covering all supported locales; add visual regression tests for locale-dependent UI elements

## F22: Wrong SQL Column Name Causes Empty Voucher Query
- **Affected Services**: Voucher-Service
- **Fault Type**: Data
- **Symptom**: "Print Voucher" button in Flow Three (Consign & Voucher) returns "Empty. No data!" despite valid voucher records existing; HTTP 200 with empty result set; Prometheus `voucher_query_empty_total` increments on valid order IDs
- **Root Cause**: The dynamically constructed SQL `SELECT` statement references a column name that does not exist in the table specified in the `FROM` clause. The database returns an empty result set (or an error caught and converted to empty) instead of the expected voucher data
- **Diagnostic Steps**:
  1. Check `voucher_query_empty_total` and `sql_error_total{service="voucher-service",type="column_not_found"}` in Prometheus
  2. Search Voucher-Service logs for `SQLSyntaxErrorException`, `Unknown column`, or `Column not found` errors; inspect the constructed SQL statement in debug-level logs
  3. `kubectl exec <voucher-pod> -- mysql -e "DESC voucher_table"` to list actual column names; compare against the column names in the service's query builder code: `kubectl exec <voucher-pod> -- grep -r "SELECT.*FROM.*voucher" /app/`
- **Remediation & Mitigation**: Immediate — correct the column name in the SQL query builder to match the actual table schema. Long-term — use an ORM (JPA/Hibernate) or compile-time query validation to catch column-name mismatches at build time; add integration tests that execute every query path against a test database schema
