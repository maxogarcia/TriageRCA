# Train Ticket Code-Level Faults

## TT-CODE-F22-01: Wrong SQL Column Produces Empty Voucher Result
- **Affected Services**: ts-voucher-service, ts-ui-dashboard
- **Fault Type**: Code-level
- **Symptom**: Valid voucher requests return HTTP 200 with an empty collection or an application-level `Empty. No data!` message; `voucher_query_empty_total` increases for known order IDs; Python/MySQL logs may contain `Unknown column`, `OperationalError`, or `ProgrammingError` if the data-access layer does not suppress the database error
- **Root Cause**: The F22 query builder references a column that does not exist in the selected table or uses a stale renamed column. Error handling converts the database exception into an empty result, hiding the code/schema mismatch from callers
- **Diagnostic Steps**:
  1. Compare successful voucher creation with empty-result and SQL-error counts; if custom instrumentation exports `voucher_query_empty_total` and `sql_error_total`, graph `rate(voucher_query_empty_total[5m])` and `rate(sql_error_total{service="ts-voucher-service",type="column_not_found"}[5m])`, otherwise derive both counts from structured voucher logs; select a known order ID as a control case
  2. Inspect a failed trace and structured SQL log to capture the query template and database error without exposing bound secrets; verify that the empty response follows the exception handler
  3. Using an approved read-only database account, run `kubectl exec -n <namespace> -it <voucher-db-pod> -- mysql -u <read-only-user> -p -D <database> -e 'DESCRIBE <voucher_table>;'` and compare the schema with the query mapping in the deployed `ts-voucher-service` revision; do not place the password directly in the command line
- **Remediation & Mitigation**: Immediate — deploy the corrected column mapping or roll back to the schema-compatible release and verify the known voucher control case. Long-term — replace string-built SQL with validated mappings or an ORM, run migrations and service code in compatibility tests, fail loudly on unexpected database errors, and add contract tests for every voucher query path
