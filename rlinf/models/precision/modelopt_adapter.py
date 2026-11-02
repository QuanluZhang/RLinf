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

"""Draft ModelOpt integration interfaces; distributed lifecycle is incomplete."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn


class ModelOptAdapter:
    """Isolate optional ModelOpt APIs from RL workers."""

    def prepare(
        self, model: nn.Module, recipe_path: str,
        calibration_loop: Callable[[nn.Module], None], real_forward: bool = False,
    ) -> nn.Module:
        """Calibrate an unsharded model using a provider recipe."""
        from modelopt.recipe import load_recipe
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.config import CompressConfig

        recipe = load_recipe(recipe_path)
        model = mtq.quantize(
            model, config=recipe.quantize, forward_loop=calibration_loop
        )
        if real_forward:
            mtq.compress(model, config=CompressConfig(
                compress={"default": False}, quant_gemm=True
            ))
        return model

    def search_mixed_precision(self, model: nn.Module, **kwargs: Any) -> Any:
        """Search formats with caller-supplied trajectory/loss callbacks."""
        import modelopt.torch.quantization as mtq

        return mtq.auto_quantize(model, **kwargs)

    def save_prepared(self, model: nn.Module, path: str) -> None:
        """Save provider structure and unsharded tensors."""
        import modelopt.torch.opt as mto

        mto.save(model, path)

    def restore_prepared(self, model: nn.Module, path: str) -> nn.Module:
        """Restore before FSDP wrapping or compilation."""
        import modelopt.torch.opt as mto

        return mto.restore(model, path, map_location="cpu")

    def save_structure(self, model: nn.Module, checkpoint_dir: str) -> None:
        """Save provider metadata beside distributed tensors."""
        import modelopt.torch.opt as mto

        torch.save(mto.modelopt_state(model), Path(checkpoint_dir) / "modelopt_state.pt")

    def restore_structure(self, model: nn.Module, checkpoint_dir: str) -> nn.Module:
        """Restore structure before distributed checkpoint loading."""
        import modelopt.torch.opt as mto

        return mto.restore_from_modelopt_state(
            model, modelopt_state_path=str(Path(checkpoint_dir) / "modelopt_state.pt")
        )

    def validate_training_contract(self) -> None:
        """Block activation until distributed lifecycle contracts are validated."""
        # TODO(agent): Certify FSDP, versions, scale sync and cache contracts.
        raise NotImplementedError("ModelOpt RL training is draft scaffolding only")

    def after_optimizer_step(self, model: nn.Module) -> None:
        """Invalidate caches after a confirmed successful optimizer update."""
        # TODO(agent): Select a public provider API and handle skipped updates.
        raise NotImplementedError("Cache invalidation contract is unresolved")

    def snapshot_metadata(self, model: nn.Module) -> dict[str, Any]:
        """Return recipe, scale revision and execution mode for a snapshot."""
        raise NotImplementedError("Precision snapshot metadata is unresolved")

    def finalize_snapshot(self, model: nn.Module, version: int) -> None:
        """Finalize provider state before activating a received version."""
        raise NotImplementedError("Atomic snapshot activation is unresolved")
