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

## 2026-06-14 - [XSS via Array Type Confusion / Implicit toString() Bypass]
**Vulnerability:** The sanitization functions `escapeHtml`, `escapeJs`, and `escapeRegExp` in `static/app.js` contained a type check that returned early if the input was not a string (`if (typeof unsafe !== 'string') return unsafe;`). This allowed an attacker to bypass the filter by passing an array or object containing malicious strings (e.g. `["Attacker", "<script>alert(1)</script>"]`). When the returned array was concatenated into the DOM via template literals, it was implicitly cast to a string by the browser (executing the `.toString()` method), which joined the array elements with commas and injected the unescaped malicious payload directly into the HTML.
**Learning:** Type-checking before escaping must be done carefully in languages with implicit type coercion like JavaScript. If an unescaped object bypasses the filter, the browser will likely stringify it later when rendering, executing any hidden payloads.
**Prevention:** Explicitly cast inputs to strings *before* applying escaping rules (e.g., `String(unsafe)`) instead of early-returning non-string inputs.
