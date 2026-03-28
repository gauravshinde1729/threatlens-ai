# Privilege Escalation Prevention (CWE-269)

## Overview
Privilege escalation (PrivEsc) allows a lower-privileged user or process to gain elevated permissions. Local PrivEsc is the most common post-exploitation step — a successful RCE at www-data often leads to root within minutes if defenses are absent.

## Fixes

### 1. Principle of Least Privilege
Services must run as dedicated low-privilege users with no login shell:
```bash
useradd --system --no-create-home --shell /usr/sbin/nologin appuser
```
Never run application processes as root.

### 2. RBAC and sudo Hardening
```
# /etc/sudoers — be explicit, never use NOPASSWD for broad commands
appuser ALL=(root) NOPASSWD: /usr/bin/systemctl restart myapp
```
Audit sudoers with `sudo -l`. Remove wildcard rules. Enable `requiretty`.

### 3. SUID/SGID Auditing
```bash
find / -perm /4000 -o -perm /2000 2>/dev/null
```
Remove SUID bits from binaries that don't require them. Monitor for new SUID files via auditd.

### 4. Linux Capabilities
Drop all capabilities and grant only required ones:
```python
# In Docker
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE myimage
```
Avoid `CAP_SYS_ADMIN`, `CAP_SYS_PTRACE`.

### 5. Seccomp and AppArmor/SELinux
Apply seccomp profiles to restrict available syscalls. Use AppArmor profiles to confine file and network access by path.

### 6. Kernel Hardening
```
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1
```

## Detection
Alert on: unexpected setuid/setgid calls, new cron entries by non-root users, `su`/`sudo` from unexpected users, `/etc/passwd` or `/etc/shadow` modifications.
