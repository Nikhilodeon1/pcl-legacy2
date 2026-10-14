import os
import shutil
import subprocess
import threading
import time

POLL_SEC = float(os.environ.get("POD_MONITOR_POLL_SEC", 15))
IDLE_MIN = float(os.environ.get("POD_MONITOR_IDLE_MIN", 5))
ACTIVE_MIN = float(os.environ.get("POD_MONITOR_ACTIVE_MIN", 2))
REMINDER_MIN = float(os.environ.get("POD_MONITOR_REMINDER_MIN", 10))
UTIL_PCT = float(os.environ.get("POD_MONITOR_UTIL_PCT", 10))

_SWITCH_DOWN_MSG = """
{bar}
SWITCH PODS NOW — GPU HAS BEEN IDLE FOR {mins:.0f}+ MIN (UTIL < {util:.0f}%).
THIS LOOKS LIKE DATA LOADING / PARSING / PREPROCESSING, NOT TRAINING.
YOU ARE PAYING GPU-POD PRICE FOR CPU-ONLY WORK — SWITCH TO A CHEAP POD.
{bar}
"""

_SWITCH_UP_MSG = """
{bar}
SWITCH BACK TO THE GPU POD NOW — SUSTAINED GPU ACTIVITY FOR {mins:.0f}+ MIN
(UTIL >= {util:.0f}%). REAL TRAINING HAS RESUMED.
IF YOU MOVED DOWN TO A CHEAP POD FOR THE DATA-LOAD PHASE, MOVE BACK NOW.
{bar}
"""

_BAR = "=" * 70


def _read_gpu_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return max(float(line.strip()) for line in out.stdout.strip().splitlines() if line.strip())
    except ValueError:
        return None


def _monitor_loop(poll_sec, idle_min, active_min, reminder_min, util_pct, verbose):
    idle_streak_start = None
    active_streak_start = None
    state = None
    last_alert_ts = 0.0
    consecutive_read_failures = 0

    while True:
        time.sleep(poll_sec)
        util = _read_gpu_util()
        now = time.time()

        if util is None:
            consecutive_read_failures += 1
            if consecutive_read_failures >= 3:
                if verbose:
                    print("[pod_monitor] nvidia-smi unavailable — stopping monitor.", flush=True)
                return
            continue
        consecutive_read_failures = 0

        if util >= util_pct:
            active_streak_start = active_streak_start or now
            idle_streak_start = None
        else:
            idle_streak_start = idle_streak_start or now
            active_streak_start = None

        active_dur = (now - active_streak_start) if active_streak_start else 0.0
        idle_dur = (now - idle_streak_start) if idle_streak_start else 0.0

        if active_dur >= active_min * 60 and state != "active":
            print(_SWITCH_UP_MSG.format(bar=_BAR, mins=active_min, util=util_pct), flush=True)
            state, last_alert_ts = "active", now
        elif idle_dur >= idle_min * 60 and state != "idle":
            print(_SWITCH_DOWN_MSG.format(bar=_BAR, mins=idle_min, util=util_pct), flush=True)
            state, last_alert_ts = "idle", now
        elif state is not None and (now - last_alert_ts) >= reminder_min * 60:
            msg = _SWITCH_UP_MSG if state == "active" else _SWITCH_DOWN_MSG
            elapsed_min = (active_dur if state == "active" else idle_dur) / 60
            print(msg.format(bar=_BAR, mins=elapsed_min, util=util_pct), flush=True)
            last_alert_ts = now


_started = False
_lock = threading.Lock()


def watch_pod(poll_sec=None, idle_min=None, active_min=None, reminder_min=None,
              util_pct=None, verbose=False):

    global _started
    with _lock:
        if _started:
            return
        if shutil.which("nvidia-smi") is None:
            if verbose:
                print("[pod_monitor] no nvidia-smi on this machine — monitor disabled.", flush=True)
            _started = True
            return
        t = threading.Thread(
            target=_monitor_loop,
            args=(poll_sec or POLL_SEC, idle_min or IDLE_MIN, active_min or ACTIVE_MIN,
                  reminder_min or REMINDER_MIN, util_pct or UTIL_PCT, verbose),
            daemon=True,
            name="pod-monitor",
        )
        t.start()
        _started = True
        if verbose:
            print(f"[pod_monitor] watching GPU util every {poll_sec or POLL_SEC:.0f}s "
                  f"(idle>={idle_min or IDLE_MIN:.0f}min / active>={active_min or ACTIVE_MIN:.0f}min "
                  f"triggers an alert).", flush=True)


if __name__ == "__main__":
    watch_pod(verbose=True)
    print("pod_monitor running standalone — Ctrl+C to stop.")
    while True:
        time.sleep(60)
