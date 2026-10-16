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

"""LeKiwi mobile base composition and hardware configuration."""

from dataclasses import dataclass

from ..discovery import RobotConfig
from ..parts.mobility import LeKiwiBase, MobileBase
from ..robot import Robot


class LeKiwiRobot(Robot):
    """LeKiwi wheel base composed under the single part name ``base``."""

    ROBOT_TYPE = "LeKiwi"

    @classmethod
    def build(
        cls,
        *,
        port: str,
        node_rank: int,
        worker_rank: int = 0,
        env_idx: int = 0,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> "LeKiwiRobot":
        """Compose an unconnected LeKiwi robot from its serial port."""
        base_cls = MobileBase.backend("lekiwi")
        base = base_cls(
            port,
            wheel_radius=wheel_radius,
            base_radius=base_radius,
            max_raw=max_raw,
            node_rank=node_rank,
            worker_name=f"{base_cls.__name__}-{worker_rank}-{env_idx}",
        )
        return cls(base=base)


@dataclass
class LeKiwiConfig(RobotConfig):
    """Hardware and placement configuration for a LeKiwi base."""

    port: str | None = None
    """Serial device the three-wheel servo bus is on, such as ``/dev/ttyACM0``."""

    wheel_radius: float = 0.05
    """Wheel radius in metres."""

    base_radius: float = 0.125
    """Centre-to-wheel distance in metres."""

    max_raw: int = 3000
    """Absolute per-wheel servo velocity limit."""


LeKiwiRobot.register_type(LeKiwiConfig)
