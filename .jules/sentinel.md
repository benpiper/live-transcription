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

## 2026-05-31 - [DOM XSS bypass via implicit string conversion]
**Vulnerability:** The sanitization functions `escapeHtml`, `escapeJs`, and `escapeRegExp` in `static/app.js` returned the input unmodified if it was not of type `string`. This allowed arrays or objects to bypass the sanitization, leading to an XSS vulnerability when these values were implicitly converted to strings (e.g., via `toString()`) during template literal interpolation.
**Learning:** Type checking alone is insufficient for sanitization if the input can later be implicitly cast to a string by the Javascript engine during rendering.
**Prevention:** Explicitly handle `null` or `undefined` inputs, and forcibly cast all other inputs to a string before performing string replacements to ensure sanitization applies uniformly.
