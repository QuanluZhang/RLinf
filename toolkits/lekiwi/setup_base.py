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

try:
    for name, motor in bus.motors.items():
        input(
            f"\nPower off, connect ONLY the {name} wheel motor, "
            "restore power, then press Enter: "
        )
        bus.setup_motor(name)
        print(f"{name}: ID {motor.id}, baud rate 1000000")
finally:
    if bus.is_connected:
        # Other wheel motors are disconnected during individual setup.
        bus.disconnect(disable_torque=False)
