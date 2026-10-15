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

"""Stream a USB webcam to a remote receiver over TCP.

Runs on the machine with the camera (the Raspberry Pi). Frames are read with
OpenCV, encoded as JPEG, and sent as length-prefixed messages so the receiver can
tell one frame from the next. Frame rate is paced to ``--fps`` so the recording
plays back in real time.

Usage::

    # On the Pi, with the camera attached:
    python webcam_sender.py --host <mac-ip>
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import socket
import struct
import time


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


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


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    """Send one JPEG frame as a 4-byte big-endian length prefix plus the bytes."""
    sock.sendall(struct.pack(">I", len(payload)))
    sock.sendall(payload)


def run_sender(args: argparse.Namespace, logger: logging.Logger) -> int:
    import cv2

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {args.device}")
    sock = None
    frames = 0
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera %d opened at %dx%d; streaming to %s:%d",
            args.device,
            width,
            height,
            args.host,
            args.port,
        )

        sock = _connect(args.host, args.port, args.connect_timeout)
        interval = 1.0 / args.fps
        next_tick = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to read a frame; stopping.")
                break
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
            )
            if not ok:
                logger.warning("Failed to encode a frame; skipping.")
                continue
            _send_frame(sock, encoded.tobytes())
            frames += 1
            next_tick += interval
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        logger.info("Streaming interrupted.")
    finally:
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()
        cap.release()
    logger.info("Sent %d frames.", frames)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Receiver hostname or IP")
    parser.add_argument(
        "--port", type=int, default=5556, help="Receiver port (default: %(default)s)"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Capture width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Capture height (default: 480)"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Stream frame rate (default: 30)"
    )
    parser.add_argument(
        "--quality", type=int, default=90, help="JPEG quality 1..100 (default: 90)"
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Seconds to keep retrying the connection (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and greater than zero")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be in 1..100")
    if not math.isfinite(args.connect_timeout) or args.connect_timeout <= 0:
        parser.error("--connect-timeout must be finite and greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = _get_logger()
    try:
        return run_sender(args, logger)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 0
    except Exception:
        logger.exception("Webcam sender failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
