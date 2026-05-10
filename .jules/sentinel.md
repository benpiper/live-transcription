## 2024-05-01 - [XSS via DOM innerHTML]
**Vulnerability:** Several places in the frontend dashboard (`static/app.js`) were rendering user-controlled content like transcription text and speaker IDs by directly inserting them into the DOM using `innerHTML` and template literals without any HTML escaping.
**Learning:** Even internal or backend-generated data (like AI-transcribed text or config watchwords) must be treated as untrusted user input before rendering into the DOM, as spoken text could be crafted to include script tags resulting in a Cross-Site Scripting (XSS) vulnerability.
**Prevention:** Always use an HTML escaping function (like `escapeHtml` or DOM elements' `textContent` properties) when dynamically constructing HTML content that includes potentially user-controlled data.

## 2024-05-02 - [Missing and Overly Permissive CORS Configuration]
**Vulnerability:** The application was missing an explicit Cross-Origin Resource Sharing (CORS) configuration in the FastAPI backend (`web_server.py`), meaning cross-origin frontend requests (such as from Render to an on-premise deployment) would fail. Adding CORS configuration securely is required, as allowing all origins (`["*"]`) leads to an overly permissive CORS configuration.
**Learning:** For a split deployment architecture, CORS must be explicitly configured to only permit trusted frontend domains.
**Prevention:** Use an environment variable like `FRONTEND_URL` to define and inject the allowed origins dynamically without resorting to wildcard (`*`) matching, and provide safe fallback origins (like localhost) for local development environments.
## 2026-05-10 - [Regex Injection / ReDoS Vulnerability]
**Vulnerability:** The `highlightWatchwords` function in `static/app.js` was constructing a `RegExp` object using unescaped watchword strings. If a user added a watchword containing special regex characters, it could lead to regex injection, potentially causing unexpected behavior or a Regular Expression Denial of Service (ReDoS) attack in the client browser.
**Learning:** Any user-provided or dynamically generated string must be escaped before being passed to the `RegExp` constructor.
**Prevention:** Implement and use an `escapeRegExp` function to sanitize strings by escaping special characters (e.g., using `replace(/[.*+?^${}()|[\]\\]/g, '\\$&')`) before interpolation into a regular expression.
