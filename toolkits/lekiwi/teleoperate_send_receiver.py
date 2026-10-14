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

"""Remote keyboard teleoperation of LeKiwi wheel motors.

``teleoperate_base.py`` needs the keyboard physically attached to the machine that
drives the motors, because pynput listens to the local X11 input. Over SSH the
receiver has no local keyboard, so this module splits the two halves across a TCP
link:

* ``receiver`` runs on the machine wired to the motors and reuses the kinematics
  and motor loop from ``teleoperate_base.py``.
* ``sender`` runs on the controlling machine, captures key-down / key-up events
  with pynput, and streams them to the receiver.

The wire protocol is newline-delimited text: ``down <key>``, ``up <key>``, and a
``ping`` heartbeat. A receiver whose sender falls silent stops the base, so a
stalled controlling machine cannot leave the base moving.

Usage (start the receiver first)::

    # On the Pi, wired to the motors:
    python teleoperate_send_receiver.py receiver --port /dev/ttyACM0

    # On the Mac, where the keyboard is:
    python teleoperate_send_receiver.py sender --host <pi-ip>
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import signal
import socket
import threading
import time

from teleoperate_base import (
    SPEED_NAMES,
    WHEEL_IDS,
    KeyboardControl,
    _get_logger,
    _handle_sigterm,
    body_to_wheel_raw,
    make_bus,
    stop_and_disconnect,
)

DEFAULT_PORT = 5555
HEARTBEAT_INTERVAL = 0.5
_DOWN = "down"
_UP = "up"
_PING = "ping"


class SocketKeySource:
    """Held keys fed by a remote sender over TCP.

    A daemon thread accepts sender connections and parses ``down`` / ``up`` /
    ``ping`` lines into a held-key set. :meth:`snapshot` reports the connection as
    live only while heartbeats arrive, so a dead or stalled sender reads as
    disconnected and the control loop stops the base.
    """

    def __init__(self, host: str, port: int, heartbeat_timeout: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self._pressed: set[str] = set()
        self._conn: socket.socket | None = None
        self._last_msg = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._serve, name="key-receiver", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            conn = self._conn
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[bool, set[str]]:
        """Return ``(connected, held_keys)`` atomically.

        ``connected`` is true only while a sender is attached and its heartbeats
        are recent.
        """
        with self._lock:
            connected = self._conn is not None and (
                time.monotonic() - self._last_msg <= self.heartbeat_timeout
            )
            return connected, set(self._pressed)

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.5)  # let the loop notice stop() promptly
            while not self._stop.is_set():
                try:
                    conn, _addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._handle_conn(conn)

    def _handle_conn(self, conn: socket.socket) -> None:
        with self._lock:
            self._conn = conn
            self._pressed.clear()
            self._last_msg = time.monotonic()
        conn.settimeout(0.5)
        buffer = b""
        try:
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buffer += data
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    self._handle_line(line.decode("utf-8", errors="ignore"))
        finally:
            with contextlib.suppress(OSError):
                conn.close()
            with self._lock:
                self._conn = None
                self._pressed.clear()

    def _handle_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        event, _, key = line.partition(" ")
        with self._lock:
            self._last_msg = time.monotonic()
            if event == _PING:
                return
            if event == _DOWN:
                self._pressed.add(key)
            elif event == _UP:
                self._pressed.discard(key)


def run_receiver(
    source: SocketKeySource,
    bus,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Drive the base from remote key events until Q/Esc, sender loss, or interrupt."""
    control = KeyboardControl()
    previous_command = None
    was_connected = False
    try:
        if bus is not None:
            bus.connect()
            bus.disable_torque()
            for name in WHEEL_IDS:
                bus.write("Operating_Mode", name, 1)
                bus.write("Acceleration", name, 10)
                bus.write("Goal_Velocity", name, 0, normalize=False)
            bus.enable_torque()

        logger.info(
            "Receiver listening on %s:%d. Start the sender on the controlling "
            "machine. W/A/S/D move, Z/X turn, R/F change speed, Q or Esc quits.",
            source.host,
            source.port,
        )
        while True:
            connected, keys = source.snapshot()
            if connected and not was_connected:
                logger.info("Sender connected.")
            elif was_connected and not connected:
                logger.warning("Sender lost; stopping the base.")
                if bus is not None:
                    try:
                        bus.sync_write(
                            "Goal_Velocity", dict.fromkeys(WHEEL_IDS, 0), normalize=False
                        )
                    except Exception:
                        logger.exception("Could not stop the base after sender loss")
            was_connected = connected

            if not connected:
                time.sleep(0.1)
                continue

            if "q" in keys or "esc" in keys:
                break

            started = time.monotonic()
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
        if bus is not None and bus.is_connected:
            stop_and_disconnect(bus, logger)


def run_sender(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Capture keys with pynput and stream press/release events to the receiver."""
    from pynput import keyboard

    sock = _connect(args.host, args.port, args.connect_timeout)
    stop_event = threading.Event()
    send_lock = threading.Lock()

    def send(payload: str) -> None:
        """Send one line, marking the session stopped on any socket error."""
        try:
            with send_lock:
                sock.sendall((payload + "\n").encode("utf-8"))
        except OSError:
            stop_event.set()

    def key_name(key) -> str | None:
        char = getattr(key, "char", None)
        if char and char.isprintable() and not char.isspace():
            return char.lower()
        if key == keyboard.Key.esc:
            return "esc"
        return None

    def on_press(key) -> None:
        name = key_name(key)
        if name is not None:
            send(f"{_DOWN} {name}")

    def on_release(key) -> None:
        name = key_name(key)
        if name is not None:
            send(f"{_UP} {name}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_INTERVAL):
            send(_PING)

    threading.Thread(target=heartbeat, name="key-heartbeat", daemon=True).start()

    logger.info("Forwarding keys to %s:%d. Press Ctrl+C to stop.", args.host, args.port)
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        listener.stop()
        with contextlib.suppress(OSError):
            sock.close()


def _connect(host: str, port: int, timeout: float) -> socket.socket:
    """Open a TCP connection to the receiver, retrying until ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=min(timeout, 1.0))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise ConnectionError(
                    f"Could not connect to receiver at {host}:{port}: {exc}. "
                    "Start the receiver first."
                ) from exc
            time.sleep(0.5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the sender/receiver subcommands and validate their options."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sender = sub.add_parser("sender", help="Run on the controlling machine")
    sender.add_argument("--host", required=True, help="Receiver hostname or IP")
    sender.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Receiver port (default: %(default)s)"
    )
    sender.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Seconds to keep retrying the connection (default: %(default)s)",
    )

    receiver = sub.add_parser("receiver", help="Run on the machine wired to the motors")
    receiver.add_argument(
        "--port", help="USB servo controller port; see lerobot-find-port"
    )
    receiver.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without opening a motor port",
    )
    receiver.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="Interface to bind (default: %(default)s)",
    )
    receiver.add_argument(
        "--listen-port", type=int, default=DEFAULT_PORT, help="Port to listen on (default: %(default)s)"
    )
    receiver.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=2.0,
        help="Stop if no message from the sender for this many seconds (default: %(default)s)",
    )
    receiver.add_argument(
        "--fps", type=float, default=30.0, help="Control frequency (default: 30 Hz)"
    )
    receiver.add_argument(
        "--wheel-radius", type=float, default=0.05, help="Wheel radius in metres"
    )
    receiver.add_argument(
        "--base-radius",
        type=float,
        default=0.125,
        help="Centre-to-wheel distance in metres",
    )
    receiver.add_argument(
        "--max-raw",
        type=int,
        default=3000,
        help="Absolute servo velocity limit (default: 3000)",
    )

    args = parser.parse_args(argv)

    if args.command == "receiver":
        if not args.dry_run and not args.port:
            parser.error("--port is required unless --dry-run is used")
        for name in ("fps", "wheel_radius", "base_radius", "heartbeat_timeout"):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                parser.error(
                    f"--{name.replace('_', '-')} must be finite and greater than zero"
                )
        if not 1 <= args.max_raw <= 32767:
            parser.error("--max-raw must be in 1..32767")
        if not 1 <= args.listen_port <= 65535:
            parser.error("--listen-port must be in 1..65535")
    else:
        if not 1 <= args.port <= 65535:
            parser.error("--port must be in 1..65535")
        if not math.isfinite(args.connect_timeout) or args.connect_timeout <= 0:
            parser.error("--connect-timeout must be finite and greater than zero")

    return args


def _run_sender_main(args: argparse.Namespace, logger: logging.Logger) -> int:
    try:
        run_sender(args, logger)
    except KeyboardInterrupt:
        logger.info("Sender stopped.")
    except Exception:
        logger.exception("Sender failed")
        return 1
    return 0


def _run_receiver_main(args: argparse.Namespace, logger: logging.Logger) -> int:
    source = SocketKeySource(args.listen_host, args.listen_port, args.heartbeat_timeout)
    source.start()
    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        bus = None if args.dry_run else make_bus(args.port)
        run_receiver(source, bus, args, logger)
    except KeyboardInterrupt:
        logger.info("Teleoperation interrupted; motor cleanup attempted.")
    except Exception:
        logger.exception("Receiver failed")
        return 1
    finally:
        source.stop()
        signal.signal(signal.SIGTERM, previous_handler)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = _get_logger()
    if args.command == "sender":
        return _run_sender_main(args, logger)
    return _run_receiver_main(args, logger)


if __name__ == "__main__":
    raise SystemExit(main())
