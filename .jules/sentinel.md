## 2024-05-01 - [XSS via DOM innerHTML]
**Vulnerability:** Several places in the frontend dashboard (`static/app.js`) were rendering user-controlled content like transcription text and speaker IDs by directly inserting them into the DOM using `innerHTML` and template literals without any HTML escaping.
**Learning:** Even internal or backend-generated data (like AI-transcribed text or config watchwords) must be treated as untrusted user input before rendering into the DOM, as spoken text could be crafted to include script tags resulting in a Cross-Site Scripting (XSS) vulnerability.
**Prevention:** Always use an HTML escaping function (like `escapeHtml` or DOM elements' `textContent` properties) when dynamically constructing HTML content that includes potentially user-controlled data.

## 2024-05-02 - [Missing and Overly Permissive CORS Configuration]
**Vulnerability:** The application was missing an explicit Cross-Origin Resource Sharing (CORS) configuration in the FastAPI backend (`web_server.py`), meaning cross-origin frontend requests (such as from Render to an on-premise deployment) would fail. Adding CORS configuration securely is required, as allowing all origins (`["*"]`) leads to an overly permissive CORS configuration.
**Learning:** For a split deployment architecture, CORS must be explicitly configured to only permit trusted frontend domains.
**Prevention:** Use an environment variable like `FRONTEND_URL` to define and inject the allowed origins dynamically without resorting to wildcard (`*`) matching, and provide safe fallback origins (like localhost) for local development environments.

## 2025-03-05 - [Regex Injection / ReDoS via Watchwords]
**Vulnerability:** User-controlled watchwords were directly interpolated into a `new RegExp()` constructor in `static/app.js` without sanitization. This allowed users to input regex special characters, leading to Regex Injection and potential Regular Expression Denial of Service (ReDoS) which could crash or slow down the frontend.
**Learning:** Any user-provided strings used to construct regular expressions dynamically must be sanitized.
**Prevention:** Always use an `escapeRegExp` function (like `string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')`) on user input before passing it to the `RegExp` constructor.

## 2024-05-03 - [XSS Bypass via Array Payloads]
**Vulnerability:** The sanitization functions (`escapeHtml`, `escapeJs`) previously used a `typeof !== 'string'` check that returned the unescaped input. Malicious array or object payloads could bypass this check and be executed upon implicit string conversion in the DOM.
**Learning:** Type checks that simply return the original non-string input are insufficient because arrays/objects are implicitly cast to strings when rendered, bypassing the filter.
**Prevention:** Always check for null values explicitly and cast inputs to strings (e.g., `String(unsafe)`) before applying escaping logic to prevent type confusion vulnerabilities.

## 2026-09-07 - [Insecure SSL/TLS Configuration in Email Alert]
**Vulnerability:** The email alert script (`email_alert.py`) disabled SSL/TLS certificate verification globally by default when connecting to the transcription feed (both HTTP initial context fetch and WebSocket connection). It hardcoded `ssl_context.check_hostname = False` and `ssl_context.verify_mode = ssl.CERT_NONE`. This allowed for potential man-in-the-middle (MITM) attacks where an attacker could intercept and potentially modify the connection.
**Learning:** Default configurations for SSL/TLS must always be secure (e.g. using `ssl.create_default_context()`). Disabling certificate verification should only be allowed as an explicit opt-in configuration for specific use cases (like local development or specific self-signed setups).
**Prevention:** Always verify SSL/TLS certificates by default. Require explicit configuration (e.g. an `insecure_skip_verify` flag) to bypass validation and log a clear warning when such a configuration is used.
