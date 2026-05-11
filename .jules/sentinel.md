## 2024-05-01 - [XSS via DOM innerHTML]
**Vulnerability:** Several places in the frontend dashboard (`static/app.js`) were rendering user-controlled content like transcription text and speaker IDs by directly inserting them into the DOM using `innerHTML` and template literals without any HTML escaping.
**Learning:** Even internal or backend-generated data (like AI-transcribed text or config watchwords) must be treated as untrusted user input before rendering into the DOM, as spoken text could be crafted to include script tags resulting in a Cross-Site Scripting (XSS) vulnerability.
**Prevention:** Always use an HTML escaping function (like `escapeHtml` or DOM elements' `textContent` properties) when dynamically constructing HTML content that includes potentially user-controlled data.

## 2024-05-02 - [Missing and Overly Permissive CORS Configuration]
**Vulnerability:** The application was missing an explicit Cross-Origin Resource Sharing (CORS) configuration in the FastAPI backend (`web_server.py`), meaning cross-origin frontend requests (such as from Render to an on-premise deployment) would fail. Adding CORS configuration securely is required, as allowing all origins (`["*"]`) leads to an overly permissive CORS configuration.
**Learning:** For a split deployment architecture, CORS must be explicitly configured to only permit trusted frontend domains.
**Prevention:** Use an environment variable like `FRONTEND_URL` to define and inject the allowed origins dynamically without resorting to wildcard (`*`) matching, and provide safe fallback origins (like localhost) for local development environments.

## 2026-05-11 - [Regex Injection / ReDoS via unescaped string in RegExp]
**Vulnerability:** The `highlightWatchwords` function in `static/app.js` was passing unescaped user-controlled input (`watchwords`) into a `new RegExp()` constructor. This allowed special characters to alter the regex logic, which could lead to Regex Injection or Regular Expression Denial of Service (ReDoS) if a malicious user provided a crafted watchword pattern (like `(((a.*)+)+)+`).
**Learning:** Even simple string interpolation into a regex constructor is a potential injection vector if the string originates from user input or storage.
**Prevention:** Always escape regex-specific characters using a dedicated escaping function (like `escapeRegExp`) before interpolating dynamic strings into a `new RegExp()` constructor.
