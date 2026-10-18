import numpy as np

from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.robotics.parts.cameras.webcam import WebcamCamera

print("discovered:", WebcamCamera.discover())   # e.g. {"0"} — use this as serial_number

camera = WebcamCamera(CameraInfo(
    name="webcam", serial_number="0", camera_type="webcam",
    resolution=(640, 480), fps=30,
))
camera.connect()
try:
    print("ready:", camera.is_ready(timeout=3.0))
    frame = camera.get_observation()["frame"]
    print("frame shape:", frame.shape, "dtype:", frame.dtype)
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
finally:
    camera.disconnect()
print("webcam OK")
