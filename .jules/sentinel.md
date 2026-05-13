## 2024-05-01 - [XSS via DOM innerHTML]
**Vulnerability:** Several places in the frontend dashboard (`static/app.js`) were rendering user-controlled content like transcription text and speaker IDs by directly inserting them into the DOM using `innerHTML` and template literals without any HTML escaping.
**Learning:** Even internal or backend-generated data (like AI-transcribed text or config watchwords) must be treated as untrusted user input before rendering into the DOM, as spoken text could be crafted to include script tags resulting in a Cross-Site Scripting (XSS) vulnerability.
**Prevention:** Always use an HTML escaping function (like `escapeHtml` or DOM elements' `textContent` properties) when dynamically constructing HTML content that includes potentially user-controlled data.

## 2024-05-02 - [Missing and Overly Permissive CORS Configuration]
**Vulnerability:** The application was missing an explicit Cross-Origin Resource Sharing (CORS) configuration in the FastAPI backend (`web_server.py`), meaning cross-origin frontend requests (such as from Render to an on-premise deployment) would fail. Adding CORS configuration securely is required, as allowing all origins (`["*"]`) leads to an overly permissive CORS configuration.
**Learning:** For a split deployment architecture, CORS must be explicitly configured to only permit trusted frontend domains.
**Prevention:** Use an environment variable like `FRONTEND_URL` to define and inject the allowed origins dynamically without resorting to wildcard (`*`) matching, and provide safe fallback origins (like localhost) for local development environments.

## 2025-02-28 - [High] Prevent ReDoS via user-controlled regex
**Vulnerability:** User-controlled strings (watchwords) were directly interpolated into a `new RegExp()` constructor in `static/app.js` without proper sanitization. This allowed users to inject special characters or malformed regex patterns, opening up a potential Regular Expression Denial of Service (ReDoS) vulnerability.
**Learning:** In vanilla JavaScript, dynamically creating regular expressions from user input (like search keywords or watchwords) is dangerous. Since there's no built-in regex sanitization standard library in JavaScript, custom implementation of string sanitization is critical before creating instances of `RegExp`.
**Prevention:** Always sanitize user-controlled strings using a function like `escapeRegExp` (e.g. `string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')`) before passing them to the `new RegExp()` constructor.
