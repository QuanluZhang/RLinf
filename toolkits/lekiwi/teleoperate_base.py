# Copyright 2026 The RLinf Authors.
# Copyright 2024-2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keyboard teleoperation of LeKiwi wheel motors over a local USB connection.

Adapted from LeRobot's examples/lekiwi/teleoperate.py, the keyboard mapping in
LeKiwiClient, and LeKiwi._body_to_wheel_raw (Apache-2.0).
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import time
from collections.abc import Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop

WHEEL_IDS = {"left": 7, "back": 8, "right": 9}
SPEED_LEVELS = ((0.1, 30.0), (0.2, 60.0), (0.3, 90.0))
SPEED_NAMES = ("slow", "medium", "fast")


class KeyboardControl:
    """Map held keys to chassis velocity, changing speed once per key press."""

    def __init__(self) -> None:
        self.speed_index = 0
        self._previous: set[str] = set()

    def action(self, keys: Collection[str]) -> tuple[float, float, float]:
        """Return forward/lateral velocity in m/s and rotation in degrees/s."""
        keys = set(keys)
        pressed = keys - self._previous
        self._previous = keys
        if "r" in pressed:
            self.speed_index = min(self.speed_index + 1, len(SPEED_LEVELS) - 1)
        if "f" in pressed:
            self.speed_index = max(self.speed_index - 1, 0)
        xy, theta = SPEED_LEVELS[self.speed_index]
        return (
            xy * (("w" in keys) - ("s" in keys)),
            xy * (("a" in keys) - ("d" in keys)),
            theta * (("z" in keys) - ("x" in keys)),
        )


def body_to_wheel_raw(
    x: float,
    y: float,
    theta: float,
    wheel_radius: float = 0.05,
    base_radius: float = 0.125,
    max_raw: int = 3000,
) -> dict[str, int]:
    """Convert chassis m/s and degrees/s to signed servo velocity units.

    Wheel order, mounting angles, and proportional limiting match LeRobot.
    Radii are in metres; base_radius is the centre-to-wheel distance.
    """
    if not all(math.isfinite(v) for v in (x, y, theta, wheel_radius, base_radius)):
        raise ValueError("Velocities and radii must be finite numbers.")
    if wheel_radius <= 0 or base_radius <= 0 or not 1 <= max_raw <= 32767:
        raise ValueError("Radii must be positive and max_raw must be in 1..32767.")
    rotation = math.radians(theta) * base_radius
    raw = [
        (
            math.cos(math.radians(angle)) * x
            + math.sin(math.radians(angle)) * y
            + rotation
        )
        / wheel_radius
        * (4096.0 / (2.0 * math.pi))
        for angle in (150, -90, 30)
    ]
    scale = max(1.0, max(abs(value) for value in raw) / max_raw)
    return {
        name: round(value / scale) for name, value in zip(WHEEL_IDS, raw, strict=True)
    }


def make_bus(port: str) -> FeetechMotorsBus:
    """Create a bus containing only the three already configured wheel IDs."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus(
        port=port,
        motors={
            name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)
            for name, motor_id in WHEEL_IDS.items()
        },
    )


def stop_and_disconnect(bus: FeetechMotorsBus, logger: logging.Logger) -> None:
    """Attempt a stop and torque disable on every wheel, then close USB."""
    try:
        # Stop reachable wheels before waiting for any missing wheel's reply.
        try:
            bus.sync_write(
                "Goal_Velocity", dict.fromkeys(WHEEL_IDS, 0), normalize=False
            )
        except Exception:
            logger.exception("Could not send the initial wheel stop")
        for name in WHEEL_IDS:
            try:
                bus.write("Goal_Velocity", name, 0, normalize=False, num_retry=2)
            except Exception:
                logger.exception("Could not send a stop to the %s wheel", name)
        for name in WHEEL_IDS:
            try:
                bus.disable_torque(name, num_retry=2)
            except Exception:
                logger.exception("Could not disable torque on the %s wheel", name)
    finally:
        bus.disconnect(disable_torque=False)


def run_teleoperation(
    keyboard: KeyboardTeleop,
    bus: FeetechMotorsBus | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Run until Q, Esc, keyboard loss, or an exception; always attempt cleanup.

    Passing bus=None previews commands without connecting to any motors.
    """
    control = KeyboardControl()
    previous_command = None
    try:
        keyboard.connect()
        if not keyboard.is_connected:
            raise RuntimeError(
                "Keyboard capture is unavailable. On macOS allow your terminal in "
                "Privacy & Security > Accessibility / Input Monitoring and restart it. "
                "On Linux use a local X11 desktop session."
            )
        if bus is not None:
            bus.connect()
            bus.disable_torque()
            for name in WHEEL_IDS:
                bus.write("Operating_Mode", name, 1)
                bus.write("Acceleration", name, 10)
                bus.write("Goal_Velocity", name, 0, normalize=False)
            bus.enable_torque()

        logger.info(
            "Ready at slow speed. Hold W/A/S/D to move, Z/X to turn. "
            "R/F change speed; Q, Esc or Ctrl+C quit. Release movement keys to stop."
        )
        while keyboard.is_connected:
            started = time.monotonic()
            keys = {
                key.lower() for key in keyboard.get_action() if isinstance(key, str)
            }
            if "q" in keys or not keyboard.is_connected:
                break
            previous_speed = control.speed_index
            command = body_to_wheel_raw(
                *control.action(keys),
                wheel_radius=args.wheel_radius,
                base_radius=args.base_radius,
                max_raw=args.max_raw,
            )
            if control.speed_index != previous_speed:
                logger.info("Speed: %s", SPEED_NAMES[control.speed_index])
            if bus is not None:
                bus.sync_write("Goal_Velocity", command, normalize=False)
            elif command != previous_command:
                logger.info("Preview wheel velocities: %s", command)
            previous_command = command
            time.sleep(max(1.0 / args.fps - (time.monotonic() - started), 0.0))
    finally:
        try:
            if bus is not None and bus.is_connected:
                stop_and_disconnect(bus, logger)
        finally:
            if keyboard.is_connected:
                keyboard.disconnect()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate options before opening the keyboard or motor port."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", help="USB servo controller port; see lerobot-find-port"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview keys without opening a motor port",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Control frequency (default: 30 Hz)"
    )
    parser.add_argument(
        "--wheel-radius", type=float, default=0.05, help="Wheel radius in metres"
    )
    parser.add_argument(
        "--base-radius",
        type=float,
        default=0.125,
        help="Centre-to-wheel distance in metres",
    )
    parser.add_argument(
        "--max-raw",
        type=int,
        default=3000,
        help="Absolute servo velocity limit (default: 3000)",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and not args.port:
        parser.error("--port is required unless --dry-run is used")
    for name in ("fps", "wheel_radius", "base_radius"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and greater than zero"
            )
    if not 1 <= args.max_raw <= 32767:
        parser.error("--max-raw must be in 1..32767")
    return args


def _get_logger() -> logging.Logger:
    try:
        from rlinf.utils.logging import get_logger

        return get_logger()
    except ImportError:
        # Standalone LeRobot environments do not need RLinf's training dependencies.
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(handler)
        logger.propagate = False
        return logger


def _handle_sigterm(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    """Start keyboard teleoperation on the computer connected to the controller."""
    args = parse_args(argv)
    logger = _get_logger()
    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        from lerobot.teleoperators.keyboard.teleop_keyboard import (
            KeyboardTeleop,
            KeyboardTeleopConfig,
        )

        keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="lekiwi_base_keyboard"))
        bus = None if args.dry_run else make_bus(args.port)
        run_teleoperation(keyboard, bus, args, logger)
    except KeyboardInterrupt:
        logger.info("Teleoperation interrupted; motor cleanup attempted.")
    except Exception:
        logger.exception("Base teleoperation failed")
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
