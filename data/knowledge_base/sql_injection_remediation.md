# SQL Injection Remediation Guide (CWE-89)

## Overview
SQL injection allows attackers to manipulate database queries by injecting malicious SQL syntax through user-controlled input. Consequences include data exfiltration, authentication bypass, and in some configs, RCE via `xp_cmdshell`.

## Fixes

### 1. Parameterized Queries (Mandatory)
Never concatenate user input into SQL strings:
```python
# VULNERABLE
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# SAFE
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

### 2. ORM Usage
Django ORM, SQLAlchemy, and similar frameworks parameterize by default. Avoid `.raw()` and `text()` with unsanitized input.

### 3. Stored Procedures
Stored procedures are safe only when they themselves use parameterized queries internally. A stored procedure that concatenates strings is still vulnerable.

### 4. Input Validation
Validate type and range (e.g., `user_id` must be a positive integer). Reject early before reaching the database layer.

### 5. Database Hardening
- Application DB account should have minimum required privileges (SELECT only for read paths, no DROP/CREATE).
- Disable `xp_cmdshell` on MSSQL. Disable `LOAD DATA INFILE` on MySQL if unused.
- Enable database audit logging for anomalous query volumes.

### 6. Error Handling
Never expose raw database errors to clients. Generic error messages prevent schema enumeration.

## Detection
WAF rules for: `UNION SELECT`, `1=1`, `--`, `'OR'`, `SLEEP(`, `BENCHMARK(`. Monitor for unusually large result sets or high query latency from specific parameters.
