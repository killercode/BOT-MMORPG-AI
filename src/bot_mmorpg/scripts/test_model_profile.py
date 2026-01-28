import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

try:
    from .directkeys import PressKey, ReleaseKey
    from .grabscreen import grab_screen
    from .models import inception_v3 as googlenet
except ImportError:
    from directkeys import PressKey, ReleaseKey
    from grabscreen import grab_screen
    from models import inception_v3 as googlenet


KEYCODES = {
    "W": 0x11,
    "A": 0x1E,
    "S": 0x1F,
    "D": 0x20,
    "Q": 0x10,
    "E": 0x12,
    "R": 0x13,
    "F": 0x21,
    "C": 0x2E,
    "V": 0x2F,
    "TAB": 0x0F,
    "SPACE": 0x39,
    "SHIFT": 0x2A,
    "CTRL": 0x1D,
    "ALT": 0x38,
    "1": 0x31,
}


DEFAULT_WEIGHTS = np.array(
    [
        4.5,
        0.1,
        0.1,
        0.1,
        1.8,
        1.8,
        0.5,
        0.5,
        0.2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
)


def _load_profile(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "movement_keys" not in data:
        raise ValueError("Profile must define 'movement_keys'.")
    return data


def _resolve_keycode(value: str) -> int:
    if value.startswith("0x"):
        return int(value, 16)
    if value.isdigit():
        return int(value)
    key = value.upper()
    if key not in KEYCODES:
        raise ValueError(f"Unsupported key '{value}'. Add it to KEYCODES or use a scancode.")
    return KEYCODES[key]


def _press_keys(keys: Iterable[str], hold_time: float = 0.05) -> None:
    keycodes = [_resolve_keycode(key) for key in keys]
    for keycode in keycodes:
        PressKey(keycode)
    time.sleep(hold_time)
    for keycode in keycodes:
        ReleaseKey(keycode)


def _set_movement(active: List[str], all_keys: Iterable[str]) -> None:
    for key in all_keys:
        ReleaseKey(_resolve_keycode(key))
    for key in active:
        PressKey(_resolve_keycode(key))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="BOT MMORPG - Profiled PC Runner")
    parser.add_argument("--model", default="artifacts/model/mmorpg_bot", help="Path to model file")
    parser.add_argument(
        "--profile",
        default="configs/elden_ring_nightreign_pc.json",
        help="Path to profile JSON",
    )
    parser.add_argument("--action-interval", type=float, default=0.2, help="Seconds between actions")
    args = parser.parse_args(argv)

    profile = _load_profile(Path(args.profile))
    region = tuple(profile.get("screen_region", [0, 40, 1920, 1120]))
    model_input = profile.get("model_input", {"width": 480, "height": 270})
    width = model_input["width"]
    height = model_input["height"]
    weights = np.array(profile.get("weights", DEFAULT_WEIGHTS))

    movement_keys = profile["movement_keys"]
    movement_map = {
        0: [movement_keys["forward"]],
        1: [movement_keys["back"]],
        2: [movement_keys["left"]],
        3: [movement_keys["right"]],
        4: [movement_keys["forward"], movement_keys["left"]],
        5: [movement_keys["forward"], movement_keys["right"]],
        6: [movement_keys["back"], movement_keys["left"]],
        7: [movement_keys["back"], movement_keys["right"]],
    }
    all_movement_keys = [
        movement_keys["forward"],
        movement_keys["back"],
        movement_keys["left"],
        movement_keys["right"],
    ]

    press_actions = {int(key): value for key, value in profile.get("press_actions", {}).items()}

    print(f"[profiled] Loading model from {args.model}")
    model = googlenet(width, height, 3, 1e-3, output=29)
    try:
        model.load(args.model)
        print("[profiled] Model loaded successfully.")
    except Exception as exc:
        print(f"[profiled] Failed to load model: {exc}")
        return 1

    print("[profiled] Starting bot in 3 seconds...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    last_action_time = 0.0
    last_choice: Optional[int] = None

    while True:
        frame = grab_screen(region=region)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        prediction = model.predict([frame.reshape(width, height, 3)])[0]
        weighted = np.array(prediction) * weights
        choice = int(np.argmax(weighted))

        now = time.time()
        if choice != last_choice or now - last_action_time >= args.action_interval:
            if choice in movement_map:
                _set_movement(movement_map[choice], all_movement_keys)
            elif choice in press_actions:
                _press_keys(press_actions[choice])
            else:
                _set_movement([], all_movement_keys)
            last_choice = choice
            last_action_time = now

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
