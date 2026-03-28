# Cross-Site Scripting (XSS) Prevention Guide (CWE-79)

## Overview
XSS allows attackers to inject malicious scripts into pages viewed by other users. Stored XSS persists in the database; reflected XSS is delivered via crafted URLs; DOM-based XSS manipulates the DOM without server involvement.

## Fixes

### 1. Output Encoding (Context-Aware)
Encode output based on where it lands in the HTML document:
- **HTML body**: HTML-encode (`&`, `<`, `>`, `"`, `'`)
- **JavaScript context**: JS-encode or use `json_encode()`
- **URL parameter**: URL-encode
- **CSS context**: CSS-encode or avoid dynamic CSS entirely

Use libraries: Python `markupsafe`, Java OWASP Java Encoder, React JSX auto-escaping.

### 2. Content Security Policy (CSP)
```
Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-{random}'; object-src 'none';
```
Use nonces for inline scripts. Avoid `'unsafe-inline'` and `'unsafe-eval'`.

### 3. DOM Sanitization
When innerHTML assignment is unavoidable, use DOMPurify:
```javascript
element.innerHTML = DOMPurify.sanitize(userContent);
```

### 4. Cookie Security Flags
```
Set-Cookie: session=abc; HttpOnly; Secure; SameSite=Strict
```
`HttpOnly` prevents JS access to session cookies even if XSS succeeds.

### 5. Trusted Types API
Enable via CSP header: `require-trusted-types-for 'script'`. Forces all DOM sink assignments through a validated factory.

## Detection
Audit all locations where user data reaches `innerHTML`, `document.write()`, `eval()`, or server-side template rendering. Use SAST (Semgrep, CodeQL) and DAST (OWASP ZAP).
