## 2024-05-01 - [XSS via DOM innerHTML]
**Vulnerability:** Several places in the frontend dashboard (`static/app.js`) were rendering user-controlled content like transcription text and speaker IDs by directly inserting them into the DOM using `innerHTML` and template literals without any HTML escaping.
**Learning:** Even internal or backend-generated data (like AI-transcribed text or config watchwords) must be treated as untrusted user input before rendering into the DOM, as spoken text could be crafted to include script tags resulting in a Cross-Site Scripting (XSS) vulnerability.
**Prevention:** Always use an HTML escaping function (like `escapeHtml` or DOM elements' `textContent` properties) when dynamically constructing HTML content that includes potentially user-controlled data.

## 2024-05-02 - [Missing and Overly Permissive CORS Configuration]
**Vulnerability:** The application was missing an explicit Cross-Origin Resource Sharing (CORS) configuration in the FastAPI backend (`web_server.py`), meaning cross-origin frontend requests (such as from Render to an on-premise deployment) would fail. Adding CORS configuration securely is required, as allowing all origins (`["*"]`) leads to an overly permissive CORS configuration.
**Learning:** For a split deployment architecture, CORS must be explicitly configured to only permit trusted frontend domains.
**Prevention:** Use an environment variable like `FRONTEND_URL` to define and inject the allowed origins dynamically without resorting to wildcard (`*`) matching, and provide safe fallback origins (like localhost) for local development environments.

## 2024-05-06 - [ReDoS Vulnerability in Watchword Highlighting]
**Vulnerability:** A Regular Expression Denial of Service (ReDoS) vulnerability existed in the frontend (`static/app.js`) because user-controlled watchwords were directly interpolated into a `RegExp` constructor without being escaped. An attacker or user could input a watchword like `(a+)+b` which, when matched against a long string of `a`s, would cause catastrophic backtracking, freezing the browser.
**Learning:** Even entirely client-side user input stored in `localStorage` can cause self-inflicted Denial of Service or be weaponized via CSRF/XSS vectors if fed raw into regex evaluation.
**Prevention:** Always sanitize/escape user-controlled input before passing it to `new RegExp()`. Use a function like `string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')` to escape regex tokens, in addition to escaping HTML where necessary.
