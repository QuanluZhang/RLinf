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

"""Receive a webcam stream over TCP and save it locally.

Runs on the Mac. Listens for a sender, decodes each length-prefixed JPEG frame,
and writes them into a video file. With ``--save-frames`` it also keeps the
original JPEG bytes as individual images.

Usage::

    # On the Mac:
    python webcam_receiver.py --output recording.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import socket
import struct


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _recv_all(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes, or ``None`` on a clean EOF at a frame boundary."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            if not data:
                return None
            raise ConnectionError("Connection closed mid-frame")
        data += chunk
    return bytes(data)


def _recv_frame(sock: socket.socket) -> bytes | None:
    header = _recv_all(sock, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    return _recv_all(sock, length)


def run_receiver(args: argparse.Namespace, logger: logging.Logger) -> int:
    import cv2
    import numpy as np

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    logger.info("Listening on %s:%d; waiting for a sender ...", args.host, args.port)

    conn, addr = server.accept()
    logger.info("Sender connected from %s:%d", addr[0], addr[1])

    if args.save_frames is not None:
        os.makedirs(args.save_frames, exist_ok=True)

    writer = None
    frames = 0
    try:
        while True:
            payload = _recv_frame(conn)
            if payload is None:
                break
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Dropping an undecodable frame.")
                continue
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Cannot open {args.output!r} for writing; check that --fourcc "
                        f"({args.fourcc}) matches the file extension."
                    )
                logger.info(
                    "Recording %dx%d at %.1f fps to %s",
                    width,
                    height,
                    args.fps,
                    args.output,
                )
            writer.write(frame)
            if args.save_frames is not None:
                with open(
                    os.path.join(args.save_frames, f"frame_{frames:06d}.jpg"), "wb"
                ) as fh:
                    fh.write(payload)
            frames += 1
    except KeyboardInterrupt:
        logger.info("Recording interrupted.")
    finally:
        with contextlib.suppress(OSError):
            conn.close()
        server.close()
        if writer is not None:
            writer.release()
    logger.info("Saved %d frames.", frames)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default="0.0.0.0", help="Interface to bind (default: %(default)s)"
    )
    parser.add_argument(
        "--port", type=int, default=5556, help="Port to listen on (default: %(default)s)"
    )
    parser.add_argument(
        "--output",
        default="webcam_recording.mp4",
        help="Output video file (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate written into the video (default: 30)",
    )
    parser.add_argument(
        "--fourcc",
        default="mp4v",
        help="OpenCV fourcc codec; match it to --output's extension (default: mp4v)",
    )
    parser.add_argument(
        "--save-frames",
        metavar="DIR",
        help="Also save each received JPEG frame into DIR",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and greater than zero")
    if len(args.fourcc) != 4:
        parser.error("--fourcc must be exactly four characters, e.g. mp4v, avc1, MJPG")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = _get_logger()
    try:
        return run_receiver(args, logger)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 0
    except Exception:
        logger.exception("Webcam receiver failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
