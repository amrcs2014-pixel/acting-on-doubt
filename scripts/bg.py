"""Run one command unattended:  python scripts/bg.py <logfile> <python args...>

Sets PYTHONPATH and AOD_IGNORE_CTRL=1 (console Ctrl+C/Break are ignored by this process and all children,
so a sleep/wake cycle cannot kill the run), appends the command's output to <logfile> and waits.
Launch it detached, e.g. via Win32_Process.Create.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.update(PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8", AOD_IGNORE_CTRL="1")
from aod.exp import ignore_console_ctrl   # noqa: E402

if __name__ == "__main__":
    ignore_console_ctrl()
    try:                                       # do not idle-sleep while this job runs (released on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    log, args = sys.argv[1], sys.argv[2:]
    with open(os.path.join(ROOT, log), "a", encoding="utf8") as out:
        sys.exit(subprocess.run([sys.executable] + args, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode)
