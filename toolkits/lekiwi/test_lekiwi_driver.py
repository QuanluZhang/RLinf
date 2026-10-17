import argparse
import time

import numpy as np

from rlinf.robotics.parts.mobility import LeKiwiBase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--velocity", nargs=3, type=float, default=None,
                        help="optional trial [x y theta] in m/s, m/s, deg/s")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    base = LeKiwiBase(args.port)
    base.connect()
    try:
        print("rest velocity:", base.get_observation()["velocity"])  # ~[0,0,0]

        if args.velocity is not None:
            velocity = np.asarray(args.velocity, dtype=np.float32)
            applied = base.send_action({"velocity": velocity})
            print("applied:", applied["velocity"])
            time.sleep(args.seconds)
            print("observed:", base.get_observation()["velocity"])

        base.reset()
        print("after reset:", base.get_observation()["velocity"])
    finally:
        base.disconnect()
    print("disconnected (torque off)")


if __name__ == "__main__":
    main()
