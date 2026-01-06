# Copyright 2025 The RLinf Authors.
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

"""Training script for SFT on Behavior dataset."""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="behavior_sft_pi05"
)
def main(cfg) -> None:
    """Main training function."""
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create SFT worker group (SFT only needs actor, no env/rollout)
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPSftWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Create runner
    runner = SFTRunner(
        cfg=cfg,
        actor=actor_group,
    )

    # Initialize and run
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()

