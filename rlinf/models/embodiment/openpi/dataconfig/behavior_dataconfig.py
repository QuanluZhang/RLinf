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

"""Behavior dataset config for SFT training."""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies.b1k_policy import B1kInputs, B1kOutputs


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    """Data config for Behavior-1K dataset."""

    action_sequence_keys: tuple[str, ...] = ("action",)
    delta_action_mask: tuple[int, ...] | None = None
    subsample_action_stride: int = 1
    rearrange_action_indices: tuple[int, ...] | None = None
    model_delta_action_mask: tuple[int, ...] | None = None

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # Repack transform to match keys from Behavior dataset to model inputs
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms for Behavior dataset
        data_transforms = _transforms.Group(
            inputs=[
                B1kInputs(
                    action_dim=model_config.action_dim, model_type=model_config.model_type
                )
            ],
            outputs=[B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms
        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

