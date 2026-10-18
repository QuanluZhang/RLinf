import os

from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo, WebcamCamera
from rlinf.robotics.robots import LeKiwiRobot

serials = WebcamCamera.discover()
assert serials, (
    "no webcam found: /dev/v4l/by-id/ is empty and no index opens; "
    "check lsusb and replug the camera"
)

# discover() returns basenames on Linux; the driver opens paths or indices.
def as_device(serial: str) -> str:
    if not serial.startswith("/") and os.path.exists(f"/dev/v4l/by-id/{serial}"):
        return f"/dev/v4l/by-id/{serial}"
    return serial

devices = sorted(as_device(s) for s in serials)
serial = devices[0]
print(f"found webcams: {devices}")
print(f"using {serial}")

robot = LeKiwiRobot.build(
    port="/dev/ttyACM0",
    node_rank=0,
    cameras={
        "front": CameraInfo(
            name="front",
            serial_number=serial,
            camera_type="webcam",
            resolution=(640, 480),
            fps=30,
        )
    },
)
print(robot.describe())
robot.connect()
try:
    frame = robot.parts_of_type(BaseCamera)["front"].get_observation()["frame"]
    print("frame:", frame.shape, frame.dtype)
finally:
    robot.disconnect()
