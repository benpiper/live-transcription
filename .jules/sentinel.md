## 2024-05-01 - [XSS via DOM innerHTML]
**Vulnerability:** Several places in the frontend dashboard (`static/app.js`) were rendering user-controlled content like transcription text and speaker IDs by directly inserting them into the DOM using `innerHTML` and template literals without any HTML escaping.
**Learning:** Even internal or backend-generated data (like AI-transcribed text or config watchwords) must be treated as untrusted user input before rendering into the DOM, as spoken text could be crafted to include script tags resulting in a Cross-Site Scripting (XSS) vulnerability.
**Prevention:** Always use an HTML escaping function (like `escapeHtml` or DOM elements' `textContent` properties) when dynamically constructing HTML content that includes potentially user-controlled data.

## 2024-05-02 - [Missing and Overly Permissive CORS Configuration]
**Vulnerability:** The application was missing an explicit Cross-Origin Resource Sharing (CORS) configuration in the FastAPI backend (`web_server.py`), meaning cross-origin frontend requests (such as from Render to an on-premise deployment) would fail. Adding CORS configuration securely is required, as allowing all origins (`["*"]`) leads to an overly permissive CORS configuration.
**Learning:** For a split deployment architecture, CORS must be explicitly configured to only permit trusted frontend domains.
**Prevention:** Use an environment variable like `FRONTEND_URL` to define and inject the allowed origins dynamically without resorting to wildcard (`*`) matching, and provide safe fallback origins (like localhost) for local development environments.

## 2024-05-24 - Regex Injection via Unescaped User Input
**Vulnerability:** A Denial of Service (ReDoS) and Regex Injection vulnerability existed in the frontend `static/app.js` file. The `watchwords` feature interpolated user-controlled input directly into a `new RegExp()` constructor without sanitization within the `highlightWatchwords` function.
**Learning:** This codebase dynamically creates regex patterns from user inputs (like settings or watchwords). When building dynamic regular expressions, any characters with special meaning in regex must be escaped to prevent users from altering the regex structure or passing excessively complex patterns (e.g., `(.*a){10000}`) that cause catastrophic backtracking.
**Prevention:** Always use an `escapeRegExp` helper function (like `string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')`) on any user-controlled string before interpolating it into a `new RegExp()` constructor.
