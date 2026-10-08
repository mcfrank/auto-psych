"""Agents without internet access (PI decision 2026-10-08): no network for anything
an agent runs, while opencode itself still reaches its model API.

Sherlock allows no network namespaces, so the sandbox cannot cut the network.
Instead:

* every shell command an opencode agent runs goes through ``$SHELL``, which
  the RSA spawner points at a ``bash`` wrapper (`write_shell_wrapper`). The
  wrapper runs this file, which installs a seccomp filter refusing IPv4 and
  IPv6 sockets (``socket(AF_INET|AF_INET6, ...)`` fails with EACCES) and
  then execs the real bash. A filter is inherited by every child and cannot
  be removed, so curl, pip, git, Python and anything else the agent starts
  have no internet; Unix sockets and files work as before;
* opencode's own web tools (webfetch, websearch, codesearch) are denied
  through ``OPENCODE_PERMISSION`` (`DENIED_TOOLS`); user rules override the
  built-in subagents' defaults;
* after each agent, `web_tool_uses` reads its log, and the spawner raises if
  any web tool call completed: a harness bug, not something to carry on past.

This file runs standalone (``python -I no_network.py <bash> [args...]``,
stdlib only): the wrapper calls it by path from wherever the agent's shell
starts. x86-64 Linux only; anything else raises.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import struct
import sys
from pathlib import Path

DENIED_TOOLS = ("webfetch", "websearch", "codesearch")

# seccomp / BPF constants (linux/seccomp.h, linux/filter.h, linux/audit.h)
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_KILL = 0x00000000
_SECCOMP_RET_ERRNO = 0x00050000
_EACCES = 13
_AUDIT_ARCH_X86_64 = 0xC000003E
_NR_SOCKET_X86_64 = 41
_X32_SYSCALL_BIT = 0x40000000
_AF_INET, _AF_INET6 = 2, 10
_BPF_LD_W_ABS = 0x20  # BPF_LD | BPF_W | BPF_ABS
_BPF_JEQ_K = 0x15  # BPF_JMP | BPF_JEQ | BPF_K
_BPF_JGE_K = 0x35  # BPF_JMP | BPF_JGE | BPF_K
_BPF_RET_K = 0x06  # BPF_RET | BPF_K
# struct seccomp_data: int nr; __u32 arch; __u64 instruction_pointer; __u64 args[6]
_OFF_NR, _OFF_ARCH, _OFF_ARG0 = 0, 4, 16


def _insn(code: int, jt: int, jf: int, k: int) -> bytes:
    return struct.pack("HBBI", code, jt, jf, k)


def filter_program() -> bytes:
    """BPF: on x86-64, socket(AF_INET or AF_INET6, ...) fails with EACCES;
    every other syscall is allowed; another ABI (i386, x32) kills the process."""
    deny = _SECCOMP_RET_ERRNO | _EACCES
    prog = [
        _insn(_BPF_LD_W_ABS, 0, 0, _OFF_ARCH),
        _insn(_BPF_JEQ_K, 1, 0, _AUDIT_ARCH_X86_64),
        _insn(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL),
        _insn(_BPF_LD_W_ABS, 0, 0, _OFF_NR),
        _insn(_BPF_JGE_K, 0, 1, _X32_SYSCALL_BIT),
        _insn(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL),
        _insn(_BPF_JEQ_K, 0, 4, _NR_SOCKET_X86_64),
        _insn(_BPF_LD_W_ABS, 0, 0, _OFF_ARG0),
        _insn(_BPF_JEQ_K, 1, 0, _AF_INET),
        _insn(_BPF_JEQ_K, 0, 1, _AF_INET6),
        _insn(_BPF_RET_K, 0, 0, deny),
        _insn(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW),
    ]
    return b"".join(prog)


def apply_inet_filter() -> None:
    """Install the filter on this process (and so on everything it execs)."""
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError(f"the no-network filter supports x86-64 Linux only, not {sys.platform}/{platform.machine()}")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    code = filter_program()
    buf = ctypes.create_string_buffer(code)

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(len(code) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.addressof(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed: the kernel refused the no-network filter")


def write_shell_wrapper(directory: Path, python: str, bash: str) -> Path:
    """``<directory>/bash``: runs this file's filter, then the real ``bash``.

    The wrapper is named ``bash`` because opencode accepts ``$SHELL`` only by
    a known shell name. This file is copied beside it, so the wrapper works
    from any working directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    helper = directory / "no_network.py"
    helper.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    if Path(bash).resolve() == (directory / "bash").resolve():
        raise ValueError(f"{bash} is the wrapper itself; pass the real bash")
    wrapper = directory / "bash"
    wrapper.write_text(f'#!/bin/sh\nexec "{python}" -I "{helper}" "{bash}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


_PROBE = (
    "import socket\n"
    "for fam in (socket.AF_INET, socket.AF_INET6):\n"
    "    try:\n"
    "        socket.socket(fam, socket.SOCK_STREAM).close()\n"
    "        print('OPEN', int(fam))\n"
    "    except OSError:\n"
    "        pass\n"
    "print('PROBE-DONE')\n"
)


def verify_wrapper(wrapper: Path, python: str) -> None:
    """Run the wrapper on this machine: its shell must work and refuse
    internet sockets. Raises otherwise (e.g. a kernel without seccomp)."""
    import subprocess

    probe = Path(wrapper).with_name("probe.py")
    probe.write_text(_PROBE, encoding="utf-8")
    out = subprocess.run([str(wrapper), "-c", f'"{python}" -I "{probe}"'], capture_output=True, text=True, timeout=120)
    if out.returncode != 0 or "PROBE-DONE" not in out.stdout:
        raise RuntimeError(f"the agents' no-network shell {wrapper} does not run here: {out.stderr.strip()[-500:]}")
    if "OPEN" in out.stdout:
        raise RuntimeError(f"the agents' no-network shell {wrapper} left internet sockets open: {out.stdout.strip()}")


def opencode_permission(existing: str | None) -> str:
    """``OPENCODE_PERMISSION`` with the web tools denied (merged over ``existing``)."""
    permission = json.loads(existing or "{}")
    for tool in DENIED_TOOLS:
        permission[tool] = "deny"
    return json.dumps(permission)


def web_tool_uses(log_path: Path) -> list[str]:
    """Web tool calls that completed in an opencode JSON log (none expected)."""
    used = []
    path = Path(log_path)
    if not path.exists():
        return used
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"tool_use"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part") or {}
        state = part.get("state") or {}
        if part.get("tool") in DENIED_TOOLS and state.get("status") == "completed":
            used.append(f"{part['tool']}: {json.dumps(state.get('input', {}))[:200]}")
    return used


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit("usage: no_network.py <program> [args...]")
    apply_inet_filter()
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main(sys.argv[1:])
