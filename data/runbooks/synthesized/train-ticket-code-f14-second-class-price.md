# Train Ticket Code-Level Faults

## TT-CODE-F14-01: Incorrect Second-Class Ticket Price Calculation
- **Affected Services**: ts-price-service, ts-basic-service, ts-ticketinfo-service, ts-travel-service
- **Fault Type**: Code-level
- **Symptom**: Second-class quotes differ from the expected fare table while first-class results remain correct; `ticket_price_calculation_error_total{seat_class="second"}` or price-deviation alerts rise without HTTP failures; order traces carry an internally consistent but incorrect amount
- **Root Cause**: The F14 second-class calculation branch applies an incorrect arithmetic rule, constant, multiplier, or field mapping while producing a syntactically valid amount. Normal availability checks therefore do not detect the business-logic error
- **Diagnostic Steps**:
  1. Compare second-class price distributions with first-class and a versioned fare-table baseline; if custom instrumentation exports `ticket_price_calculation_error_total{seat_class="second"}`, use it as supporting evidence, otherwise derive the mismatch count from structured quote-validation logs; identify the first release with deviation
  2. Inspect a quote trace for route, seat class, base fare, multiplier, discount inputs, and final amount; replay the same sanitized inputs through the known-good formula and compare each calculation stage
  3. Run `kubectl logs -n <namespace> deploy/ts-price-service --since=15m | grep '<trace-id>'` and, from an approved debug pod, call `http://ts-price-service:16579/api/v1/priceservice/prices/<route-id>/<train-type>` for a controlled route to compare the returned second-class fare with the approved fixture
- **Remediation & Mitigation**: Immediate — disable sale of affected second-class quotes or roll back the pricing release, recalculate impacted unpaid orders, and follow the approved correction process for completed purchases. Long-term — centralize fare rules, use table-driven unit and property tests across classes and discounts, version rule inputs, add price sanity bounds, and require golden-fixture comparison before deployment
