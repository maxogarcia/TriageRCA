# Logic Faults

## F9: RTL Display Error in Login Block CSS
- **Affected Services**: UI-Frontend
- **Fault Type**: Code-level
- **Symptom**: Login form elements (labels, inputs, buttons) are misaligned or overlapping when the UI is switched to a Right-To-Left (RTL) language (e.g., Arabic, Hebrew); no HTTP errors; visual corruption only; user reports or UI regression test failures
- **Root Cause**: The CSS for the Login Block uses hardcoded `left`/`right` positioning and directional properties (e.g., `margin-left`, `text-align: left`) instead of logical properties (e.g., `margin-inline-start`, `text-align: start`). When the `dir="rtl"` attribute is applied, the layout does not mirror correctly, producing overlapping or misplaced elements
- **Diagnostic Steps**:
  1. No Prometheus metric anomaly expected — this is a purely visual bug; check frontend error-tracking tools (e.g., Sentry) for layout-related JavaScript errors or user-reported UI issues filtered by locale
  2. Inspect the Login Block HTML in browser DevTools with `dir="rtl"` set on the `<html>` element; look for CSS rules using physical properties (`left`, `right`, `margin-left`, `padding-right`) that should be logical
  3. `kubectl exec <frontend-pod> -- grep -rn "margin-left\|padding-right\|text-align: left\|float: left" /app/src/components/Login/` to identify hardcoded directional CSS properties
- **Remediation & Mitigation**: Immediate — replace physical CSS properties with CSS Logical Properties (`margin-inline-start`, `padding-inline-end`, `text-align: start`) in the Login Block stylesheet. Long-term — adopt a CSS linting rule (e.g., `stylelint-no-physical-properties`) across the project to enforce logical properties; add visual regression tests for RTL layout in CI

## F10: Wrong API Invocation in Specific Reservation Scenario
- **Affected Services**: Contacts-Service, Reservation-Service
- **Fault Type**: Code-level
- **Symptom**: Ticket reservation fails in a specific edge case (e.g., a particular contact type or booking path); HTTP 500 or business-logic error returned; Prometheus `reservation_failures_total{reason="contact_api_error"}` increments for a subset of reservation attempts
- **Root Cause**: Contacts-Service exposes two separate APIs for the reservation flow, each handling a different business scenario. The routing logic in Reservation-Service does not correctly distinguish between the two scenarios, invoking the wrong Contacts API for a specific edge case. This produces an unexpected response that the reservation flow cannot process
- **Diagnostic Steps**:
  1. Check `reservation_failures_total{reason="contact_api_error"}` in Prometheus; filter by request attributes (contact type, booking path) to identify the failing subset
  2. Inspect Jaeger traces for failed reservations: compare the Contacts-Service API endpoint called in successful vs. failed traces; search Reservation-Service logs for the request payload that triggers the wrong API branch
  3. `kubectl logs -l app=reservation-service --tail=500 | grep -E "contacts.*api\|api.*contacts"` and `kubectl logs -l app=contacts-service --tail=500 | grep "404\|405\|unexpected"` to identify the misrouted API call
- **Remediation & Mitigation**: Immediate — fix the conditional logic in Reservation-Service that selects the Contacts API endpoint; add explicit validation of the scenario type before dispatching. Long-term — consolidate the two Contacts APIs into a single endpoint with a discriminator parameter; add contract tests between Reservation-Service and Contacts-Service to catch routing regressions

## F14: Incorrect Price Calculation for Second-Class Seats
- **Affected Services**: Price-Service, Order-Service
- **Fault Type**: Code-level
- **Symptom**: Second-class seat ticket prices are incorrect (higher or lower than expected); no HTTP errors; Prometheus metric `ticket_price_calculated{seat_class="second"}` shows values outside the expected range
- **Root Cause**: The price-calculation logic for second-class seats contains an arithmetic error — a locked/discount product is incorrectly included (or excluded) in the CPI-based price formula, producing a wrong final price. The bug exists only in the second-class calculation branch; other seat classes compute correctly
- **Diagnostic Steps**:
  1. Check `ticket_price_calculated{seat_class="second"}` histogram in Prometheus; compare mean values against expected price tables; check `price_calculation_error_total` for anomalies
  2. Search Price-Service logs for price-calculation debug output on second-class requests; trace the calculation step-by-step in Jaeger spans if calculation stages are instrumented
  3. `kubectl exec <price-pod> -- curl localhost:8080/api/v1/priceservice/prices/<route-id>` and manually verify the second-class price against the expected formula; `kubectl exec <price-pod> -- grep -n "secondClass\|SECOND\|seat_class" /app/src/main/java/price/service/PriceServiceImpl.java`
- **Remediation & Mitigation**: Immediate — correct the arithmetic formula in the second-class price calculation branch; exclude locked products from the base price computation. Long-term — add comprehensive unit tests for every seat-class pricing path with known expected outputs; implement a price-validation sanity check that flags prices deviating more than 10% from historical averages

## F21: Missing Accessibility Label on Verification Code Input
- **Affected Services**: UI-Frontend
- **Fault Type**: Code-level
- **Symptom**: Screen reader (JAWS) reads raw input value "abcd" for the Verification Code field instead of announcing the field name; Email and Password fields are read correctly; accessibility audit tools (axe, Lighthouse) flag missing `aria-labelledby` on the verification input
- **Root Cause**: The Login module adds `aria-labelledby` attributes to the Email and Password input fields, linking them to their visible labels. The Verification Code input is missing this attribute, so JAWS falls back to reading the input's default/placeholder value instead of announcing "Verification Code", confusing visually impaired users
- **Diagnostic Steps**:
  1. No Prometheus metric expected — this is an accessibility compliance issue; run Lighthouse or axe-core audit on the login page and check for `aria-labelledby` violations
  2. Inspect the Login component's HTML source for the Verification Code `<input>` element; confirm it lacks an `aria-labelledby` or `aria-label` attribute while Email and Password inputs have one
  3. `kubectl exec <frontend-pod> -- grep -n "aria-labelledby\|aria-label\|verif" /app/src/components/Login/Login.jsx` to compare attribute presence across all three input fields
- **Remediation & Mitigation**: Immediate — add `aria-labelledby="verification-code-label"` to the Verification Code input element, with a corresponding `id="verification-code-label"` on its visible label text. Long-term — add automated accessibility testing (axe-core) to the CI pipeline; enforce a linting rule (e.g., `jsx-a11y/label-has-associated-control`) that requires all form inputs to have accessible labels
