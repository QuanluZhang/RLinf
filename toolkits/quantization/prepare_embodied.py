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

# MODELOPT-PROPOSAL P2: preparation call-site sketch only; no runnable script.
#
# from rlinf.models import get_model
# from rlinf.models.precision.modelopt_adapter import ModelOptAdapter
#
# model = get_model(cfg.actor.model).cuda()
# adapter = ModelOptAdapter()
#
# @torch.no_grad()
# def calibration_loop(model):
#     for batch in calibration_batches:
#         model.predict_action_batch(batch["obs"])
#         model(forward_inputs=batch["forward_inputs"],
#               compute_logprobs=True, compute_values=False)
#
# model = adapter.prepare(
#     model, recipe_path=cfg.precision.recipe_path,
#     calibration_loop=calibration_loop,
#     real_forward=cfg.precision.real_forward)
# adapter.save_prepared(model, cfg.precision.prepared_checkpoint)
#
# TODO(agent): Implement trajectory loading/device placement, deterministic
# calibration, supported-layer recipes and artifact validation separately.
# Batches must cover rollout observations AND saved actor denoising inputs.
# Preparation is initially single-process/unsharded, outside the Ray RL loop.
