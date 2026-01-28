import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    from .models import inception_v3 as googlenet
    from ..utils import capture_screen, keyevent, select_device, swipe, tap
except ImportError:
    from models import inception_v3 as googlenet
    from bot_mmorpg.utils import capture_screen, keyevent, select_device, swipe, tap


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
    if "movement" not in data:
        raise ValueError("Profile must define a 'movement' section.")
    return data


def _movement_vector(choice: int) -> Tuple[int, int]:
    mapping = {
        0: (0, -1),  # W
        1: (0, 1),  # S
        2: (-1, 0),  # A
        3: (1, 0),  # D
        4: (-1, -1),  # WA
        5: (1, -1),  # WD
        6: (-1, 1),  # SA
        7: (1, 1),  # SD
    }
    return mapping.get(choice, (0, 0))


def _execute_action(
    choice: int,
    movement: Dict,
    tap_actions: Dict[int, Tuple[int, int]],
    key_events: Dict[int, str],
    device,
) -> None:
    if choice in tap_actions:
        x, y = tap_actions[choice]
        tap(int(x), int(y), device=device)
        return

    if choice in key_events:
        keyevent(key_events[choice], device=device)
        return

    center_x, center_y = movement["center"]
    radius = movement.get("radius", 120)
    duration_ms = movement.get("duration_ms", 120)
    dx, dy = _movement_vector(choice)
    if dx == 0 and dy == 0:
        tap(int(center_x), int(center_y), device=device)
        return
    target_x = int(center_x + dx * radius)
    target_y = int(center_y + dy * radius)
    swipe(int(center_x), int(center_y), target_x, target_y, duration_ms=duration_ms, device=device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="BOT MMORPG - Android/ADB Runner")
    parser.add_argument("--model", default="artifacts/model/mmorpg_bot", help="Path to model file")
    parser.add_argument(
        "--profile",
        default="configs/diablo_immortal_android.json",
        help="Path to Android profile JSON",
    )
    parser.add_argument("--device", default=None, help="ADB device serial (optional)")
    parser.add_argument("--action-interval", type=float, default=0.2, help="Seconds between actions")
    args = parser.parse_args(argv)

    profile_path = Path(args.profile)
    profile = _load_profile(profile_path)

    device = select_device(args.device)

    model_input = profile.get("model_input", {"width": 480, "height": 270})
    width = model_input["width"]
    height = model_input["height"]

    print(f"[android] Loading model from {args.model}")
    model = googlenet(width, height, 3, 1e-3, output=29)
    try:
        model.load(args.model)
        print("[android] Model loaded successfully.")
    except Exception as exc:
        print(f"[android] Failed to load model: {exc}")
        return 1

    tap_actions = {
        int(key): tuple(value) for key, value in profile.get("tap_actions", {}).items()
    }
    key_events = {int(key): value for key, value in profile.get("key_events", {}).items()}

    print("[android] Starting bot in 3 seconds...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    last_action_time = 0.0
    last_choice = None

    while True:
        frame = capture_screen(device=device)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        prediction = model.predict([frame.reshape(width, height, 3)])[0]
        weighted = np.array(prediction) * DEFAULT_WEIGHTS
        choice = int(np.argmax(weighted))

        now = time.time()
        if choice != last_choice or now - last_action_time >= args.action_interval:
            _execute_action(choice, profile["movement"], tap_actions, key_events, device)
            last_choice = choice
            last_action_time = now

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
