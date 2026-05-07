## 2024-05-01 - [XSS via DOM innerHTML]
**Vulnerability:** Several places in the frontend dashboard (`static/app.js`) were rendering user-controlled content like transcription text and speaker IDs by directly inserting them into the DOM using `innerHTML` and template literals without any HTML escaping.
**Learning:** Even internal or backend-generated data (like AI-transcribed text or config watchwords) must be treated as untrusted user input before rendering into the DOM, as spoken text could be crafted to include script tags resulting in a Cross-Site Scripting (XSS) vulnerability.
**Prevention:** Always use an HTML escaping function (like `escapeHtml` or DOM elements' `textContent` properties) when dynamically constructing HTML content that includes potentially user-controlled data.

## 2024-05-02 - [Missing and Overly Permissive CORS Configuration]
**Vulnerability:** The application was missing an explicit Cross-Origin Resource Sharing (CORS) configuration in the FastAPI backend (`web_server.py`), meaning cross-origin frontend requests (such as from Render to an on-premise deployment) would fail. Adding CORS configuration securely is required, as allowing all origins (`["*"]`) leads to an overly permissive CORS configuration.
**Learning:** For a split deployment architecture, CORS must be explicitly configured to only permit trusted frontend domains.
**Prevention:** Use an environment variable like `FRONTEND_URL` to define and inject the allowed origins dynamically without resorting to wildcard (`*`) matching, and provide safe fallback origins (like localhost) for local development environments.

## 2026-05-07 - [Regex Injection / ReDoS via Unsanitized Regex Interpolation]
**Vulnerability:** The `highlightWatchwords` function in `static/app.js` was interpolating user-controlled watchwords directly into a `new RegExp()` constructor without sanitization. This allowed users to input regex control characters (like `.*`) which could lead to unexpected matching behavior or Regular Expression Denial of Service (ReDoS) if a complex, backtracking regex was provided.
**Learning:** Any user-controlled string that is going to be used as part of a regular expression must be escaped first.
**Prevention:** Use an `escapeRegExp` function (e.g., `string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')`) to escape all regex special characters before passing user input to `new RegExp()`.
