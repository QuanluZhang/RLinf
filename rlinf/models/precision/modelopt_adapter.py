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

# MODELOPT-PROPOSAL P1: interface sketch only. Every line is a comment;
# this file defines no class, imports no dependency and executes no code.
# See proposals/modelopt_embodied_integration.md for call sites and open work.
#
# Proposed imports (only this adapter imports ModelOpt):
# from modelopt.recipe import load_recipe
# import modelopt.torch.opt as mto
# import modelopt.torch.quantization as mtq
# from modelopt.torch.quantization.config import CompressConfig
#
# class ModelOptAdapter:
#     def prepare(self, model, recipe_path, calibration_loop,
#                 real_forward=False):
#         recipe = load_recipe(recipe_path)
#         model = mtq.quantize(
#             model, config=recipe.quantize, forward_loop=calibration_loop)
#         if real_forward:
#             mtq.compress(model, config=CompressConfig(
#                 compress={"default": False}, quant_gemm=True))
#         return model
#
#     def save_prepared(self, model, path):
#         mto.save(model, path)  # structure + weights/scales, unwrapped model
#
#     def restore_prepared(self, model, path):
#         return mto.restore(model, path, map_location="cpu")
#
#     def save_structure(self, model, checkpoint_dir):
#         torch.save(mto.modelopt_state(model),
#                    Path(checkpoint_dir) / "modelopt_state.pt")
#
#     def restore_structure(self, model, checkpoint_dir):
#         return mto.restore_from_modelopt_state(
#             model, modelopt_state_path=str(
#                 Path(checkpoint_dir) / "modelopt_state.pt"))
#
# Optional mixed-precision preparation replaces mtq.quantize above:
# model, search_state = mtq.auto_quantize(
#     model,
#     constraints={"effective_bits": 6.5},
#     quantization_formats=[fp8_config, nvfp4_config],
#     data_loader=trajectory_loader,
#     forward_step=trajectory_forward,
#     loss_func=trajectory_loss,
#     disabled_layers=excluded_module_patterns,
#     num_calib_steps=128,
#     num_score_steps=64,
# )
# mto.save(model, prepared_checkpoint)
#
# TODO(agent): These calls were inspected at ModelOpt d48a5450d70e; certify
# the supported public API/version tuple before implementation. Real GEMMs
# retain higher-precision backward; packing trainable weights would freeze them.
# TODO(agent): Select a public cache invalidation/finalize-snapshot contract;
# no private quantizer attributes or invented ModelOpt methods are used here.
