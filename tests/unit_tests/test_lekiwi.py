# Copyright 2026 The RLinf Authors.
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

"""Hardware-free checks for LeKiwi keyboard movement and shutdown."""

import contextlib
import io
import logging
import math
import unittest

from robot_mocks.lekiwi import FakeKeyboard, FakeWheelBus

from toolkits.lekiwi.teleoperate_base import (
    KeyboardControl,
    body_to_wheel_raw,
    parse_args,
    run_teleoperation,
)


class TestLeKiwi(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parse_args(["--port", "fake", "--fps", "1000"])
        self.logger = logging.getLogger("test_lekiwi")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False

    def test_keys_and_opposing_directions(self) -> None:
        control = KeyboardControl()
        for key, expected in {
            "w": (0.1, 0.0, 0.0),
            "s": (-0.1, 0.0, 0.0),
            "a": (0.0, 0.1, 0.0),
            "d": (0.0, -0.1, 0.0),
            "z": (0.0, 0.0, 30.0),
            "x": (0.0, 0.0, -30.0),
        }.items():
            with self.subTest(key=key):
                self.assertEqual(control.action({key}), expected)
        self.assertEqual(control.action(set("wasdzx")), (0.0, 0.0, 0.0))
        self.assertEqual(control.action(set()), (0.0, 0.0, 0.0))

    def test_speed_changes_once_per_press_and_stays_in_range(self) -> None:
        control = KeyboardControl()
        for _ in range(5):
            self.assertEqual(control.action({"r", "w"})[0], 0.2)
        control.action(set())
        self.assertEqual(control.action({"r", "w"})[0], 0.3)
        for _ in range(5):
            control.action(set())
            self.assertEqual(control.action({"r", "w"})[0], 0.3)
        for _ in range(5):
            control.action({"f"})
            control.action(set())
        self.assertEqual(control.action({"w"})[0], 0.1)

    def test_wheel_directions_and_units(self) -> None:
        self.assertEqual(
            body_to_wheel_raw(0, 0, 0), dict.fromkeys(("left", "back", "right"), 0)
        )
        self.assertEqual(
            body_to_wheel_raw(0.1, 0, 0), {"left": -1129, "back": 0, "right": 1129}
        )
        self.assertEqual(
            body_to_wheel_raw(0, 0.1, 0), {"left": 652, "back": -1304, "right": 652}
        )
        self.assertEqual(
            body_to_wheel_raw(0, 0, 30), {"left": 853, "back": 853, "right": 853}
        )
        reverse = body_to_wheel_raw(-0.1, 0, 0)
        self.assertEqual(
            reverse,
            {key: -value for key, value in body_to_wheel_raw(0.1, 0, 0).items()},
        )

    def test_wheel_limit_preserves_relative_speeds(self) -> None:
        original = body_to_wheel_raw(0.3, 0.2, 90, max_raw=32767)
        limited = body_to_wheel_raw(0.3, 0.2, 90, max_raw=100)
        peak = max(abs(value) for value in original.values())
        self.assertEqual(max(abs(value) for value in limited.values()), 100)
        for name in original:
            self.assertAlmostEqual(limited[name], original[name] * 100 / peak, delta=1)

    def test_release_and_quit_stop_wheels(self) -> None:
        keyboard = FakeKeyboard([{"W"}, set(), {"a"}, {"q", "w"}])
        bus = FakeWheelBus()
        run_teleoperation(keyboard, bus, self.args, self.logger)
        self.assertEqual(bus.commands[0], body_to_wheel_raw(0.1, 0, 0))
        self.assertFalse(any(bus.commands[1].values()))
        self.assertEqual(len(bus.commands), 4)
        self.assertFalse(any(bus.commands[-1].values()))
        self.assert_stopped(bus, keyboard)

    def test_listener_loss_stops_without_replaying_held_key(self) -> None:
        keyboard = FakeKeyboard([{"w"}, None])
        bus = FakeWheelBus()
        run_teleoperation(keyboard, bus, self.args, self.logger)
        self.assertEqual(len(bus.commands), 2)
        self.assertFalse(any(bus.commands[-1].values()))
        self.assert_stopped(bus, keyboard)

    def test_no_keyboard_never_opens_usb(self) -> None:
        keyboard = FakeKeyboard([], available=False)
        bus = FakeWheelBus()
        with self.assertRaisesRegex(RuntimeError, "Keyboard capture"):
            run_teleoperation(keyboard, bus, self.args, self.logger)
        self.assertFalse(bus.connect_attempted)

    def test_interrupt_and_input_error_stop_wheels(self) -> None:
        for error in (KeyboardInterrupt(), RuntimeError("input failure")):
            with self.subTest(error=type(error).__name__):
                keyboard = FakeKeyboard([{"z"}, error])
                bus = FakeWheelBus()
                with self.assertRaises(type(error)):
                    run_teleoperation(keyboard, bus, self.args, self.logger)
                self.assert_stopped(bus, keyboard)

    def test_movement_error_stops_and_closes_usb(self) -> None:
        keyboard = FakeKeyboard([{"w"}])
        bus = FakeWheelBus(fail_motion=True)
        with self.assertRaisesRegex(OSError, "USB error"):
            run_teleoperation(keyboard, bus, self.args, self.logger)
        self.assert_stopped(bus, keyboard)

    def test_unreachable_wheel_does_not_prevent_other_stops_or_port_close(self) -> None:
        keyboard = FakeKeyboard([{"z"}])
        bus = FakeWheelBus(fail_motion=True, lost_wheel="left")
        with self.assertRaises(OSError):
            run_teleoperation(keyboard, bus, self.args, self.logger)
        self.assertEqual(bus.stop_attempts[-3:], ["left", "back", "right"])
        self.assertEqual(bus.disable_attempts[-3:], ["left", "back", "right"])
        self.assertEqual(bus.velocity["back"], 0)
        self.assertEqual(bus.velocity["right"], 0)
        self.assertFalse(bus.is_connected)
        self.assertFalse(keyboard.is_connected)

    def test_partial_connection_failure_closes_port(self) -> None:
        keyboard = FakeKeyboard([])
        bus = FakeWheelBus(fail_connect=True)
        with self.assertRaisesRegex(OSError, "handshake"):
            run_teleoperation(keyboard, bus, self.args, self.logger)
        self.assert_stopped(bus, keyboard)

    def test_preview_runs_without_motor_bus(self) -> None:
        keyboard = FakeKeyboard([{"w"}, set(), {"q"}])
        run_teleoperation(keyboard, None, self.args, self.logger)
        self.assertFalse(keyboard.is_connected)

    def test_invalid_options_fail_before_connecting(self) -> None:
        for flags in (
            [],
            ["--dry-run", "--fps", "0"],
            ["--dry-run", "--fps", "nan"],
            ["--dry-run", "--wheel-radius", "-1"],
            ["--dry-run", "--base-radius", "inf"],
            ["--dry-run", "--max-raw", "32768"],
            ["--dry-run", "--max-raw", "0"],
        ):
            with (
                self.subTest(flags=flags),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(flags)
        self.assertTrue(parse_args(["--dry-run"]).dry_run)
        with self.assertRaises(ValueError):
            body_to_wheel_raw(math.nan, 0, 0)

    def assert_stopped(self, bus: FakeWheelBus, keyboard: FakeKeyboard) -> None:
        self.assertFalse(any(bus.velocity.values()))
        self.assertFalse(any(bus.torque.values()))
        self.assertFalse(bus.is_connected)
        self.assertFalse(keyboard.is_connected)


if __name__ == "__main__":
    unittest.main()
