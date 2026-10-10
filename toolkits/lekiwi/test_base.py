from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

bus = FeetechMotorsBus(
    port="/dev/tty.usbmodem5AA90244061",  # Use your actual port
    motors={
        "left":  Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
        "back":  Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
        "right": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
    },
)

import time

stop = {"left": 0, "back": 0, "right": 0}

try:
    bus.connect()  # Checks only the three motors defined above.
    bus.disable_torque()

    for name in bus.motors:
        bus.write("Operating_Mode", name, 1)  # Velocity mode
        bus.write("Acceleration", name, 10)
        bus.write("Goal_Velocity", name, 0, normalize=False)

    bus.enable_torque()

    # Briefly test the left wheel; the other two stay stopped.
    bus.sync_write(
        "Goal_Velocity",
        {"left": 100, "back": 0, "right": 0},
        normalize=False,
    )
    time.sleep(0.5)

    bus.sync_write(
            "Goal_Velocity",
            {"left": 0, "back": 100, "right": 0},
            normalize=False,
        )
    time.sleep(0.5)

    bus.sync_write(
            "Goal_Velocity",
            {"left": 0, "back": 0, "right": 100},
            normalize=False,
        )
    time.sleep(0.5)
    bus.sync_write("Goal_Velocity", stop, normalize=False)
    print(bus.sync_read("Present_Velocity", normalize=False))

finally:
    if bus.is_connected:
        try:
            bus.sync_write("Goal_Velocity", stop, normalize=False)
        finally:
            bus.disconnect()  # Also disables wheel torque.
