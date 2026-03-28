# Buffer Overflow Mitigation Guide (CWE-787/119/125)

## Overview
Buffer overflows occur when programs write beyond allocated memory boundaries. Stack overflows can hijack return addresses; heap overflows corrupt allocator metadata. These are the leading cause of memory-corruption CVEs and frequent targets for RCE exploits.

## Mitigations

### 1. Memory-Safe Languages
The most effective fix is migrating critical parsing/networking code to Rust, Go, or Swift. If C/C++ is unavoidable, use bounds-checking wrappers.

### 2. Compiler Protections (Enable All)
```makefile
CFLAGS += -fstack-protector-strong -D_FORTIFY_SOURCE=2
LDFLAGS += -Wl,-z,relro,-z,now -pie
```
- **Stack canaries** (`-fstack-protector-strong`): detect stack smashing before return
- **RELRO**: makes GOT read-only, preventing overwrite
- **PIE**: enables full ASLR for the binary

### 3. OS-Level Controls
- **ASLR**: Ensure enabled (`/proc/sys/kernel/randomize_va_space = 2`).
- **NX/DEP**: Non-executable stack and heap, enabled by default on modern kernels.
- **CFI**: Clang Control Flow Integrity for forward/backward edge protection.

### 4. Safe API Usage
Replace unsafe functions with bounded alternatives:
| Unsafe | Safe |
|--------|------|
| `strcpy` | `strlcpy` / `strncpy_s` |
| `gets` | `fgets` |
| `sprintf` | `snprintf` |
| `scanf("%s")` | `scanf("%128s")` |

### 5. Fuzzing
Run AFL++ or libFuzzer against all parsing code. Integrate into CI to catch regressions.

## Detection
Monitor for: SIGSEGV/SIGABRT crashes, crash dumps with overwritten return addresses, abnormal memory growth patterns. Enable core dumps in dev/staging.
