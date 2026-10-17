# Setup Lekiwi Mobile Base

## Find Port

Install lerobot first following the doc `https://huggingface.co/docs/lerobot/main/en/installation#from-source`. Note, in raspberry pi 5, lerobot v0.5.1 has been tested; in macos, lerobot v0.6.1 has been tested.

First find the port of the motor controller with `lerobot-find-port`.
https://huggingface.co/docs/lerobot/main/en/lekiwi?example=Mac#find-the-usb-ports-associated-with-motor-board

## Set motor id of the mobile base

run toolkits/lekiwi/setup_base.py to set the motor ids. **Note:** you should update the port in this script using the port found in the above setup. You should connect indivial motor one after another with the motor controller (refer to the doc xxx).

```
python setup_base.py
```

## Test the wheels

Before running the following test, you should connect the three motors together (refer to the doc xxx).

run toolkits/lekiwi/test_base.py to test the three wheels. **Note:** you should update the port in this script using the port found in the first setup.

```
python test_base.py
```

## Drive the mobile base with the keyboard

Run `teleoperate_base.py` on the computer connected to the servo controller by USB,
using the LeRobot environment that runs `test_base.py`. It controls wheel IDs 7
(left), 8 (back), and 9 (right). Finish the individual motor setup first, then
connect all three wheels. For the first movement test, raise the wheels off the
ground and keep servo power within reach.

From `toolkits/lekiwi`, check keyboard input first:

```bash
python teleoperate_base.py --dry-run
```

This logs the wheel commands while keeping the motor port closed. Press Q to exit.
Then run with your detected controller port:

```bash
python teleoperate_base.py --port /dev/tty.usbmodem5AA90244061
```

This starts local USB control at 30 Hz and the slow speed. The script uses
LeRobot's keyboard listener and the wheel-motion calculation from
[`examples/lekiwi/teleoperate.py`](https://github.com/huggingface/lerobot/blob/main/examples/lekiwi/teleoperate.py),
[`LeKiwiClient`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/lekiwi/lekiwi_client.py),
and [`LeKiwi`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/lekiwi/lekiwi.py).
The extracted version connects directly to the three wheel motors; it does not
require the arm, cameras, or the full `lekiwi_host` process.

| Key | Action |
| --- | --- |
| W / S | Forward / backward |
| A / D | Left / right |
| Z / X | Turn left / right |
| R / F | Increase / decrease speed, once per press |
| Q / Esc / Ctrl+C | Quit and attempt to stop and disable all wheels |
| Release movement keys | Command zero velocity |

| Speed | Translation | Rotation |
| --- | --- | --- |
| Slow (initial) | 0.1 m/s | 30 degrees/s |
| Medium | 0.2 m/s | 60 degrees/s |
| Fast | 0.3 m/s | 90 degrees/s |

These are requested chassis speeds from the local LeRobot version. Wheel speeds
are limited proportionally to preserve the direction of motion. Actual motion
also depends on geometry, mounting, power, and servo behavior. Use `--wheel-radius`
and `--base-radius` (metres) if your wheel radius or centre-to-wheel distance
differs from the defaults of 0.05 and 0.125. Use `--max-raw` for a lower raw wheel
velocity limit, and `--fps` to change the control frequency.

The keyboard listener is global: keep the control terminal in use and quit before
typing in other applications. On macOS, allow the application running Python in
**System Settings > Privacy & Security > Accessibility / Input Monitoring** and
restart it if keyboard capture is unavailable. On Linux, use a local X11 desktop;
this listener does not support a headless SSH session or Wayland.

The script attempts to stop and disable every wheel on exit, lost keyboard input,
and exceptions. A disconnected USB cable or terminated process can prevent stop
commands from reaching the motors; this script does not add a motor-side watchdog.
Use the servo power switch if communication is lost. Run only one motor-control
program on the USB port at a time.

## Drive the mobile base from another computer

`teleoperate_base.py` reads the keyboard on the machine that owns the USB port.
When that machine has no local keyboard — for example over SSH from a Mac to the
Raspberry Pi — use `teleoperate_send_receiver.py`, which splits the same control
loop across a TCP link:

- `receiver` runs on the machine wired to the motors (the Pi) and sends the wheel
  commands.
- `sender` runs on the machine with the keyboard (the Mac), captures the held
  W/A/S/D/Z/X keys, and streams them to the receiver.

The key mapping and speed levels are the same as `teleoperate_base.py` above.
Start the receiver first, then the sender:

```bash
# On the Pi, wired to the motors:
python teleoperate_send_receiver.py receiver --port /dev/ttyACM0

# On the Mac, where the keyboard is:
python teleoperate_send_receiver.py sender --host <pi-ip>
```

Both sides default to port `5555`; change it with `--listen-port` on the receiver
and `--port` on the sender. The receiver takes the same `--fps`, `--wheel-radius`,
`--base-radius`, and `--max-raw` options as the local script, and `--dry-run`
previews wheel commands without opening the motor port. `--connect-timeout` sets
how long the sender keeps retrying the connection.

The receiver has a heartbeat watchdog: if the sender stops sending — it is
killed, the network drops, or the Mac sleeps — the receiver zeroes the wheel
velocities and waits for a new sender. Quit the receiver with Q or Esc as usual;
stop the sender with Ctrl+C (it also exits on its own once the receiver closes).

Both machines must reach the receiver port over the network. If the Pi cannot
reach the Mac directly, tunnel the connection back through the SSH session:

```bash
# On the Mac, open a tunnel (keep it open while streaming):
ssh -R 5555:localhost:5555 myrobots@<pi-ip>

# On the Pi, point the sender at the tunnel's loopback:
python teleoperate_send_receiver.py sender --host 127.0.0.1
```

If you change the port, keep the receiver's `--listen-port`, the sender's
`--port`, and the tunnel consistent. Run only one motor-control program on the
USB port at a time.

## Verify the keyboard script without hardware

From the RLinf repository root, run:

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests/unit_tests -p test_lekiwi.py -v
```

These tests use simulated keyboard input and a fake motor bus to check movement,
speed changes, release-to-stop, and cleanup after communication failures.
