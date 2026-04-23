import re

with open('static/app.js', 'r') as f:
    content = f.read()

# Define backend URL base
header = """
// Configuration for split deployment (Render frontend -> On-prem backend)
// Ensure to update this variable when deploying
const BACKEND_URL = window.location.hostname === 'localhost' ? '' : 'https://YOUR_ON_PREM_BACKEND_URL.com';

"""

content = header + content

# Fix fetch URL
content = content.replace("fetch('/api/login',", "fetch(`${BACKEND_URL}/api/login`,")
content = content.replace("resource !== '/api/login'", "resource !== '/api/login' && !resource.endsWith('/api/login')")

# Fix wrap fetch to prefix resource with BACKEND_URL if it starts with /api
fetch_wrapper = """const originalFetch = window.fetch;
window.fetch = async function() {
    let [resource, config] = arguments;

    // Prefix relative API routes with BACKEND_URL
    if (typeof resource === 'string' && resource.startsWith('/api')) {
        resource = `${BACKEND_URL}${resource}`;
    }

    if (!config) config = {};
    if (!config.headers) config.headers = {};

    const token = getAuthToken();
    if (token) {
        config.headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await originalFetch(resource, config);
    if (response.status === 401 && !resource.endsWith('/api/login')) {
        showLoginModal();
    }
    return response;
};"""

content = re.sub(r'const originalFetch = window\.fetch;[\s\S]*?return response;\n};', fetch_wrapper, content)

# Fix connect()
connect_logic = """function connect() {
    let protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.host;

    if (BACKEND_URL) {
        const urlObj = new URL(BACKEND_URL);
        protocol = urlObj.protocol === 'https:' ? 'wss:' : 'ws:';
        host = urlObj.host;
    }

    const token = getAuthToken();
    if (!token) {
        showLoginModal();
        return;
    }
    const wsUrl = `${protocol}//${host}/ws?token=${token}`;"""

content = re.sub(r'function connect\(\) \{\n    const protocol = window\.location\.protocol === \'https:\' \? \'wss:\' : \'ws:\';\n    const token = getAuthToken\(\);\n    if \(!token\) \{\n        showLoginModal\(\);\n        return;\n    \}\n    const wsUrl = `\$\{protocol\}//\$\{window\.location\.host\}/ws\?token=\$\{token\}`;', connect_logic, content)

with open('static/app.js', 'w') as f:
    f.write(content)
