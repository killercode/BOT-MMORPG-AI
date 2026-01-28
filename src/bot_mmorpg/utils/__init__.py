"""
Utility functions for screen capture, input handling, and preprocessing.

This module provides helper functions for:
- Screen grabbing and video capture
- Keyboard and gamepad input detection
- Direct key simulation
- Image preprocessing and data augmentation
"""

from typing import List

from .android_adb import AndroidDevice, capture_screen, keyevent, list_devices, select_device
from .android_adb import swipe, tap

__all__: List[str] = [
    "AndroidDevice",
    "capture_screen",
    "keyevent",
    "list_devices",
    "select_device",
    "swipe",
    "tap",
]
