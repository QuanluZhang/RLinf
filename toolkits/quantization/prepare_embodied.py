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

"""Draft offline preparation interface; callers provide device-placed batches."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from rlinf.models.precision.modelopt_adapter import ModelOptAdapter


def prepare_embodied_policy(
    model: nn.Module, calibration_batches: Iterable[dict[str, Any]],
    recipe_path: str, prepared_checkpoint: str, real_forward: bool = False,
) -> nn.Module:
    """Calibrate rollout and actor forward paths on an unsharded policy.

    TODO(agent): Add deterministic trajectory loading and model-specific recipes.
    """
    adapter = ModelOptAdapter()

    @torch.no_grad()
    def calibration_loop(policy: nn.Module) -> None:
        for batch in calibration_batches:
            policy.predict_action_batch(batch["obs"])
            policy(
                forward_inputs=batch["forward_inputs"],
                compute_logprobs=True, compute_values=False,
            )

    model = adapter.prepare(model, recipe_path, calibration_loop, real_forward)
    adapter.save_prepared(model, prepared_checkpoint)
    return model
