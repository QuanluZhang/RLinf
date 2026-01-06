# SFT Code Transfer Summary

This document summarizes the transfer of SFT (Supervised Fine-Tuning) code from `openpi-comet` to `RLinf` for training pi0.5 models on the Behavior simulator dataset.

## Overview

The SFT code has been successfully transferred and adapted to follow RLinf's modular design. The implementation reuses existing pi0.5 model code in RLinf and integrates seamlessly with the framework's training infrastructure.

## Components Transferred

### 1. Behavior Dataset Config
**Location**: `rlinf/models/embodiment/openpi/dataconfig/behavior_dataconfig.py`

- Implements `LeRobotB1KDataConfig` following the same pattern as other dataset configs (libero, maniskill, etc.)
- Configures data transforms for Behavior-1K dataset
- Handles repack transforms, data transforms, and model transforms

### 2. Behavior Policy Transforms
**Location**: `rlinf/models/embodiment/openpi/policies/b1k_policy.py`

- Implements `B1kInputs` and `B1kOutputs` transforms
- Handles multi-camera image processing (head, left_wrist, right_wrist)
- Extracts proprioceptive state from Behavior dataset format
- Supports depth-to-point-cloud conversion and segmentation

### 3. SFT Worker
**Location**: `rlinf/workers/sft/fsdp_sft_worker.py`

- Extends `FSDPModelManager` and `Worker` base classes
- Integrates with openpi-comet's data loading infrastructure
- Uses `create_torch_behavior_data_loader` from openpi-comet
- Implements `run_training()` method that:
  - Loads batches from Behavior dataset
  - Calls model's `sft_forward` method
  - Computes flow matching loss
  - Performs gradient updates
  - Returns training metrics

### 4. SFT Runner
**Location**: `rlinf/runners/sft_runner.py`

- Orchestrates SFT training loop
- Manages:
  - Worker initialization
  - Training loop execution
  - Checkpoint saving
  - Metric logging
  - Progress tracking

### 5. Training Script
**Location**: `examples/embodiment/train_sft.py`

- Main entry point for SFT training
- Uses Hydra for configuration management
- Follows RLinf's standard training script pattern

### 6. Configuration File
**Location**: `examples/embodiment/config/behavior_sft_pi05.yaml`

- Complete configuration for SFT training on Behavior dataset
- Includes:
  - Model configuration (pi0.5)
  - Data configuration (behavior_dataset_root, tasks, fine_grained_level)
  - Training hyperparameters
  - FSDP settings

### 7. Config Registration
**Location**: `rlinf/models/embodiment/openpi/dataconfig/__init__.py`

- Added `pi05_b1k` config to `_CONFIGS` list
- Configures pi0.5 model with Behavior-1K dataset
- Sets up default training parameters (LR schedule, batch size, etc.)

## Key Design Decisions

1. **Reuse Existing Infrastructure**:
   - Uses openpi-comet's `BehaviorLeRobotDataset` and data loading code
   - Leverages existing `OpenPi0ForRLActionPrediction` model from RLinf
   - Reuses `sft_forward` method already implemented in the model

2. **Modular Architecture**:
   - Follows RLinf's worker-based architecture
   - SFT worker is separate from RL actor workers
   - Runner follows same pattern as `EmbodiedRunner` but simplified for SFT

3. **Data Loading**:
   - Uses `create_torch_behavior_data_loader` from openpi-comet
   - Configures via `get_openpi_config` helper function
   - Supports config overrides from RLinf config file

4. **Model Forward**:
   - Uses existing `sft_forward` method in `OpenPi0ForRLActionPrediction`
   - Calls `PI0Pytorch.forward()` which computes flow matching loss
   - Loss is MSE on velocity field prediction

## Usage

### Prerequisites

1. Install RLinf dependencies
2. Ensure `openpi-comet` is in PYTHONPATH (for `BehaviorLeRobotDataset` and data loaders)
3. Download Behavior dataset to `DATASETS/behavior/2025-challenge-demos`

### Running SFT Training

```bash
cd examples/embodiment
python train_sft.py \
    --config-name=behavior_sft_pi05 \
    actor.model.model_path=/path/to/pretrained/pi0.5/model \
    actor.model.openpi.behavior_dataset_root=/path/to/behavior/dataset \
    actor.model.openpi.tasks=["turning_on_radio"] \
    runner.max_steps=20000
```

### Configuration Options

Key configuration parameters in `behavior_sft_pi05.yaml`:

- `actor.model.openpi.config_name`: Config name from dataconfig (default: "pi05_b1k")
- `actor.model.openpi.behavior_dataset_root`: Path to Behavior dataset
- `actor.model.openpi.tasks`: List of tasks to train on (e.g., `["turning_on_radio"]`)
- `actor.model.openpi.fine_grained_level`: Fine-grained orchestrator level (0, 1, 2)
- `actor.model.openpi.train_expert_only`: Freeze VLM, train only expert (default: True)
- `actor.optim.lr`: Learning rate (default: 2.5e-6)
- `runner.max_steps`: Maximum training steps
- `runner.save_interval`: Checkpoint save frequency

## Model Architecture

The SFT training uses the pi0.5 model with flow matching:

- **Input**: Multi-camera RGB images, task descriptions, proprioceptive state
- **Output**: Action predictions (velocity field)
- **Loss**: MSE loss on predicted vs. target velocity vectors
- **Training**: Flow matching approach (not standard autoregressive)

The model's `sft_forward` method:
1. Takes observation and actions
2. Samples noise and timestep
3. Creates noisy actions
4. Predicts velocity field
5. Computes MSE loss between predicted and target velocity

## Integration Points

1. **Model Loading**: Uses `rlinf.models.get_model()` to load pi0.5 models
2. **Data Loading**: Integrates with openpi-comet's data loading via `create_torch_behavior_data_loader`
3. **Distributed Training**: Uses FSDP via `FSDPModelManager`
4. **Logging**: Uses RLinf's `MetricLogger` for tensorboard/wandb
5. **Checkpointing**: Follows RLinf's checkpoint format

## Files Created/Modified

### New Files:
- `rlinf/models/embodiment/openpi/dataconfig/behavior_dataconfig.py`
- `rlinf/models/embodiment/openpi/policies/b1k_policy.py`
- `rlinf/runners/sft_runner.py`
- `examples/embodiment/train_sft.py`
- `examples/embodiment/config/behavior_sft_pi05.yaml`

### Modified Files:
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py` (added pi05_b1k config)
- `rlinf/workers/sft/fsdp_sft_worker.py` (enhanced data loading and training)

## Notes

- The implementation assumes the Behavior dataset is already downloaded
- The model expects specific input formats (handled by B1kInputs transform)
- Flow matching loss is computed directly by the model's `forward` method
- Only expert layers are trained when `train_expert_only=True` (VLM is frozen)
- The worker automatically handles iterator exhaustion by resetting the data loader

## Future Improvements

1. Add support for LoRA training (currently supports full fine-tuning)
2. Add more comprehensive validation metrics
3. Support for multi-task training with different sample weights
4. Integration with RLinf's auto-placement features
5. Add support for RGBD and RGBSegmentation modalities

