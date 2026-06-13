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

## 2025-05-15 - [XSS via Array Type Bypass in Sanitizers]
**Vulnerability:** The sanitization functions (`escapeHtml`, `escapeJs`, `escapeRegExp`) in `static/app.js` used an early return `if (typeof unsafe !== 'string') return unsafe;`. This allowed an attacker to bypass sanitization by passing an array containing malicious strings (e.g., `["<script>alert(1)</script>"]`). When these arrays were later interpolated into template literals, they were implicitly joined into strings, executing the payload and causing a Cross-Site Scripting (XSS) vulnerability.
**Learning:** Checking types for early exits can create a false sense of security. JavaScript's implicit type coercion during string interpolation can weaponize non-string types that bypassed the string-specific filters.
**Prevention:** Sanitizers should aggressively cast all input to strings (e.g., `String(unsafe)`) and handle null or undefined explicitly, rather than skipping non-string inputs.
