# Command Injection Remediation Guide (CWE-77/78)

## Overview
Command injection occurs when untrusted input is passed to a system shell or interpreter. Attackers can execute arbitrary OS commands with the privileges of the vulnerable application.

## Root Cause
Concatenating user input directly into shell commands: `os.system("ping " + user_input)`.

## Fixes

### 1. Never Use Shell Interpolation
Avoid `subprocess.run(shell=True)`. Use argument arrays instead:
```python
# VULNERABLE
subprocess.run(f"convert {filename} output.png", shell=True)

# SAFE
subprocess.run(["convert", filename, "output.png"], shell=False)
```

### 2. Input Validation and Allowlisting
Only permit strictly validated values. Reject anything not matching an allowlist — do not attempt to strip dangerous characters:
```python
ALLOWED_FORMATS = {"png", "jpg", "webp"}
if user_format not in ALLOWED_FORMATS:
    raise ValueError("Invalid format")
```

### 3. Parameterized System Calls
Use library wrappers (e.g., Pillow instead of ImageMagick CLI, `boto3` instead of `aws` CLI) to eliminate shell involvement entirely.

### 4. WAF Rules
Block patterns: `; | & $ > < ` backtick `|| && %0a %0d`.

### 5. Detection Patterns
Monitor process trees for unexpected child processes (web server spawning `bash`, `sh`, `cmd.exe`). Alert on `whoami`, `id`, `net user` execution from app processes.

## Testing
Use SAST tools (Semgrep rule `subprocess-shell-true`) and dynamic fuzzing with payloads: `; id`, `$(id)`, `` `id` ``.
