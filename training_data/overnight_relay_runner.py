"""Overnight runner for relay LoRA training on Windows.

This wrapper launches many tiny relay sessions across the night, but only when
the machine looks sufficiently idle. It is designed for CPU-constrained
machines where a single long run is impractical.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_uint),
        ("BatteryFullLifeTime", ctypes.c_uint),
    ]


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint), ("dwHighDateTime", ctypes.c_uint)]


@dataclass
class RunnerConfig:
    workspace_root: Path
    relay_script: Path
    state_path: Path
    log_path: Path
    lock_path: Path
    python_executable: Path
    start_hour: int
    end_hour: int
    min_idle_seconds: int
    poll_seconds: int
    cooldown_seconds: int
    max_sessions: int
    max_prelaunch_cpu_percent: float
    stop_on_battery: bool
    dry_run: bool


def parse_args() -> RunnerConfig:
    workspace_root = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI")
    parser = argparse.ArgumentParser(description="Overnight relay runner for CPU-friendly LoRA training")
    parser.add_argument("--start-hour", type=int, default=22, help="Local hour when the overnight window starts")
    parser.add_argument("--end-hour", type=int, default=6, help="Local hour when the overnight window ends")
    parser.add_argument("--min-idle-seconds", type=int, default=900, help="Required keyboard/mouse idle time before launch")
    parser.add_argument("--poll-seconds", type=int, default=180, help="Wait time between availability checks")
    parser.add_argument("--cooldown-seconds", type=int, default=90, help="Pause after each relay session")
    parser.add_argument("--max-sessions", type=int, default=3, help="Maximum relay sessions to run in this overnight pass")
    parser.add_argument("--max-prelaunch-cpu-percent", type=float, default=35.0, help="Do not start a session if prelaunch CPU usage is above this threshold")
    parser.add_argument("--allow-battery", action="store_true", help="Allow training while the laptop is on battery")
    parser.add_argument("--dry-run", action="store_true", help="Preview decisions without launching a real relay training session")
    args = parser.parse_args()
    return RunnerConfig(
        workspace_root=workspace_root,
        relay_script=workspace_root / "training_data" / "relay_train_qwen3_lora.py",
        state_path=workspace_root / "lora_adapter_relay" / "relay_state.json",
        log_path=workspace_root / "lora_adapter_relay" / "overnight_runner.log",
        lock_path=workspace_root / "lora_adapter_relay" / "overnight_runner.lock",
        python_executable=workspace_root / ".venv" / "Scripts" / "python.exe",
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        min_idle_seconds=args.min_idle_seconds,
        poll_seconds=args.poll_seconds,
        cooldown_seconds=args.cooldown_seconds,
        max_sessions=args.max_sessions,
        max_prelaunch_cpu_percent=args.max_prelaunch_cpu_percent,
        stop_on_battery=not args.allow_battery,
        dry_run=args.dry_run,
    )


def within_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= now.hour < end_hour
    return now.hour >= start_hour or now.hour < end_hour


def get_idle_seconds() -> float:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not user32.GetLastInputInfo(ctypes.byref(info)):
        raise ctypes.WinError()
    millis = kernel32.GetTickCount() - info.dwTime
    return millis / 1000.0


def get_power_status() -> dict:
    status = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        raise ctypes.WinError()
    return {
        "on_ac_power": status.ACLineStatus == 1,
        "battery_percent": None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent),
    }


def _filetime_to_int(filetime: FILETIME) -> int:
    return (filetime.dwHighDateTime << 32) | filetime.dwLowDateTime


def get_cpu_percent(sample_seconds: float = 1.0) -> float:
    kernel32 = ctypes.windll.kernel32
    idle1, kernel1, user1 = FILETIME(), FILETIME(), FILETIME()
    idle2, kernel2, user2 = FILETIME(), FILETIME(), FILETIME()
    if not kernel32.GetSystemTimes(ctypes.byref(idle1), ctypes.byref(kernel1), ctypes.byref(user1)):
        raise ctypes.WinError()
    time.sleep(sample_seconds)
    if not kernel32.GetSystemTimes(ctypes.byref(idle2), ctypes.byref(kernel2), ctypes.byref(user2)):
        raise ctypes.WinError()

    idle_delta = _filetime_to_int(idle2) - _filetime_to_int(idle1)
    kernel_delta = _filetime_to_int(kernel2) - _filetime_to_int(kernel1)
    user_delta = _filetime_to_int(user2) - _filetime_to_int(user1)
    total = kernel_delta + user_delta
    if total <= 0:
        return 0.0
    busy = total - idle_delta
    return round((busy / total) * 100.0, 2)


def read_state(state_path: Path) -> dict | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def append_log(config: RunnerConfig, payload: dict) -> None:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        raise RuntimeError(f"Lock file already exists: {lock_path}")
    lock_path.write_text(str(os.getpid()), encoding="utf-8")


def release_lock(lock_path: Path) -> None:
    if lock_path.exists():
        lock_path.unlink()


def machine_is_available(config: RunnerConfig) -> tuple[bool, dict]:
    idle_seconds = get_idle_seconds()
    power = get_power_status()
    cpu_percent = get_cpu_percent()
    availability = {
        "idle_seconds": round(idle_seconds, 1),
        "cpu_percent": cpu_percent,
        "on_ac_power": power["on_ac_power"],
        "battery_percent": power["battery_percent"],
    }
    reasons = []
    if idle_seconds < config.min_idle_seconds:
        reasons.append(f"idle<{config.min_idle_seconds}s")
    if cpu_percent > config.max_prelaunch_cpu_percent:
        reasons.append(f"cpu>{config.max_prelaunch_cpu_percent}%")
    if config.stop_on_battery and not power["on_ac_power"]:
        reasons.append("on_battery")
    availability["reasons"] = reasons
    return not reasons, availability


def run_one_session(config: RunnerConfig) -> dict:
    env = os.environ.copy()
    if config.dry_run:
        env["FOUNDER_AI_RELAY_DRY_RUN"] = "true"

    command = [str(config.python_executable), str(config.relay_script)]
    started_at = datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(
        command,
        cwd=str(config.workspace_root),
        capture_output=True,
        text=True,
        env=env,
    )
    ended_at = datetime.now().isoformat(timespec="seconds")
    payload = {
        "started_at": started_at,
        "ended_at": ended_at,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    append_log(config, payload)
    return payload


def main() -> None:
    config = parse_args()
    acquire_lock(config.lock_path)
    sessions_run = 0

    try:
        while sessions_run < config.max_sessions:
            now = datetime.now()
            state = read_state(config.state_path)
            if state and state.get("completed"):
                print("Relay state is already marked completed. Stopping overnight runner.")
                return

            if not within_window(now, config.start_hour, config.end_hour):
                print("Current time is outside the configured overnight window. Stopping.")
                return

            available, diagnostics = machine_is_available(config)
            print(json.dumps({"time": now.isoformat(timespec="seconds"), "availability": diagnostics}, ensure_ascii=False, indent=2))

            if not available:
                if config.dry_run:
                    print("Dry-run mode: machine is not considered free enough right now.")
                    return
                time.sleep(config.poll_seconds)
                continue

            session_payload = run_one_session(config)
            print(json.dumps({"session_result": {"returncode": session_payload["returncode"]}}, ensure_ascii=False, indent=2))
            if session_payload["returncode"] != 0:
                print("Relay session failed. Check overnight_runner.log for details.")
                return

            sessions_run += 1
            state_after = read_state(config.state_path)
            if state_after and state_after.get("completed"):
                print("Relay training reached completion during this overnight pass.")
                return

            if config.dry_run:
                print("Dry-run mode: stopping after the preview relay invocation.")
                return

            if sessions_run < config.max_sessions:
                time.sleep(config.cooldown_seconds)

        print(f"Reached max sessions for this overnight pass: {config.max_sessions}")
    finally:
        release_lock(config.lock_path)


if __name__ == "__main__":
    main()
