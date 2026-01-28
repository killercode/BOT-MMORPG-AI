import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class AndroidDevice:
    serial: str


def _adb_path() -> str:
    adb_override = os.environ.get("ADB_PATH")
    if adb_override:
        return adb_override

    sdk_root = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
    if sdk_root:
        candidate = os.path.join(sdk_root, "platform-tools", "adb")
        if os.path.exists(candidate):
            return candidate

    adb_path = shutil.which("adb")
    if adb_path:
        return adb_path

    raise FileNotFoundError(
        "ADB not found. Install Android Platform Tools or set ADB_PATH/ANDROID_SDK_ROOT."
    )


def _run_adb(args: Iterable[str], device: Optional[AndroidDevice] = None) -> subprocess.CompletedProcess:
    cmd = [_adb_path()]
    if device:
        cmd.extend(["-s", device.serial])
    cmd.extend(list(args))
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def list_devices() -> List[AndroidDevice]:
    result = _run_adb(["devices"])
    devices: List[AndroidDevice] = []
    for line in result.stdout.decode("utf-8").splitlines():
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(AndroidDevice(serial=parts[0]))
    return devices


def select_device(serial: Optional[str] = None) -> AndroidDevice:
    if serial:
        return AndroidDevice(serial=serial)
    devices = list_devices()
    if not devices:
        raise RuntimeError("No Android devices detected. Ensure adb devices shows a device.")
    if len(devices) > 1:
        raise RuntimeError(
            "Multiple Android devices detected. Pass --device to select one explicitly."
        )
    return devices[0]


def capture_screen(device: Optional[AndroidDevice] = None) -> np.ndarray:
    result = _run_adb(["exec-out", "screencap", "-p"], device=device)
    img_bytes = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode screen capture from adb output.")
    return frame


def tap(x: int, y: int, device: Optional[AndroidDevice] = None) -> None:
    _run_adb(["shell", "input", "tap", str(x), str(y)], device=device)


def swipe(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration_ms: int = 120,
    device: Optional[AndroidDevice] = None,
) -> None:
    _run_adb(
        ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
        device=device,
    )


def keyevent(keycode: str, device: Optional[AndroidDevice] = None) -> None:
    _run_adb(["shell", "input", "keyevent", keycode], device=device)
