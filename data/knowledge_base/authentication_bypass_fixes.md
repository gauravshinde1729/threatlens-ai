# Authentication Bypass Fixes (CWE-287)

## Overview
Authentication bypass vulnerabilities allow attackers to access protected resources without valid credentials. Common causes: weak token validation, missing checks on alternate paths, flawed session logic, and JWT algorithm confusion.

## Fixes

### 1. Multi-Factor Authentication (MFA)
Enforce TOTP or hardware key for all privileged accounts. MFA should be enforced server-side — never trust client-side bypass flags.

### 2. JWT Validation Hardening
```python
# Always specify allowed algorithms — never accept "alg: none"
jwt.decode(token, secret, algorithms=["HS256"])
```
Verify `iss`, `aud`, `exp`, and `nbf` claims. Reject tokens with future `iat`.

### 3. Session Management
- Regenerate session ID on privilege escalation (login, sudo).
- Set short absolute and idle timeouts (e.g., 8h absolute / 30m idle).
- Invalidate server-side sessions on logout — do not rely solely on cookie deletion.
- Bind sessions to IP or user-agent for high-risk applications.

### 4. Consistent Auth Enforcement
Apply authentication middleware globally, not per-route. A single unprotected route in an admin panel is enough for bypass. Audit all routes with a coverage tool.

### 5. Credential Storage
Store passwords with bcrypt (cost factor ≥ 12), scrypt, or Argon2id. Never MD5/SHA1. Rotate secrets immediately on suspected compromise.

### 6. Account Lockout
Lock accounts after 5–10 failed attempts within 10 minutes. Use exponential backoff for API endpoints. Log all failed authentication attempts with IP and timestamp.

## Detection
Alert on: login from new geolocation, >5 failures/minute per IP, token reuse after logout, auth bypass probe patterns (`/../admin`, `?admin=true`).
