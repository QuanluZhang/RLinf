# ModelOpt embodied low-precision integration: review-only proposal

This draft places commented interface sketches at the intended integration
points so reviewers can estimate scope. It implements no low-precision feature,
adds no dependency or executable configuration, and changes no runtime behavior.
The two new Python files contain comments only and are not runnable examples.

Base: `QuanluZhang/RLinf:main` at `39a89e3f68e2` (2 November 2026).
ModelOpt API survey: [`d48a5450d70e`](https://github.com/NVIDIA/Model-Optimizer/tree/d48a5450d70e7ec4d9ea22bc6ce4d403fef37a12).

## Review map

Search for `MODELOPT-PROPOSAL` to find every sketch in the diff.

| Marker | File / method | Proposed responsibility | Complexity |
|---|---|---|---|
| P1 | [ModelOpt adapter](../rlinf/models/precision/modelopt_adapter.py) | Isolate `quantize`, `compress`, `auto_quantize`, save and restore calls. | Small API surface; medium compatibility work. |
| P2 | [Preparation toolkit](../toolkits/quantization/prepare_embodied.py) | Calibrate both robot rollout and actor forward paths, then save a prepared policy. | Medium: trajectory data and model-specific recipes. |
| M1 | [FSDP manager](../rlinf/hybrid_engines/fsdp/fsdp_model_manager.py), `setup_model_and_optimizer` | Restore structure/weights before sync-name collection, FSDP and optimizer creation. | Small hook; medium sharding validation. |
| M2/M3 | Same manager, `load_checkpoint` / `save_checkpoint` | Restore structure early; save provider metadata alongside existing distributed tensor state. | Medium: atomic save and distributed metadata extraction. |
| M4 | Same manager, `optimizer_step` | Invalidate caches only after confirmed successful updates. | Medium: provider contract and skipped-step semantics. |
| A1/A2 | [Embodied actor](../rlinf/workers/actor/embodied_fsdp_actor_worker.py), `model_provider_func` / `sync_model_to_rollout` | Resolve checkpoint precedence and publish complete versioned policy state. | Medium; snapshot protocol shared with W1/R2. |
| A3 | Same actor, `train_micro_batch` | Existing RL forward/loss/backward uses converted modules. | Small initially; step-scoped caching/recomputation is follow-up work. |
| R1/R2 | [HF rollout worker](../rlinf/workers/rollout/hf/huggingface_worker.py), `init_worker` / `sync_model_from_actor` | Restore before compilation; finalize caches/state before activating a version. | Medium synchronously; high for async/graphs. |
| W1 | [WeightSyncer](../rlinf/hybrid_engines/weight_syncer/base.py) | Provider-independent metadata envelope, exact scale dtypes and atomic activation. | Largest shared change: bucket and patch protocols both need validation. |
| C1 | [Configuration validation](../rlinf/config.py), `validate_cfg` | Validate optional precision settings without altering storage dtype semantics. | Small validation surface; medium capability matrix. |

This is eight review files: five existing files with comment-only annotations,
two new comment-only interface sketches, and this proposal. A working feature
would also require recipes, installation/Docker support, tests and user docs;
those are deliberately outside this review-only diff.

## Actual ModelOpt calls and ordering

```text
Preparation (P2 -> P1, unsharded):
  get_model -> load_recipe -> mtq.quantize(calibration_loop)
  -> optional mtq.compress(unpacked trainable weights) -> mto.save

Actor initialization (M1):
  get_model -> mto.restore(prepared policy)
  -> collect_param_names_need_sync -> FSDP wrapping -> optimizer

Rollout initialization (R1):
  get_model -> mto.restore(the same prepared policy)
  -> eval -> optional compilation/graph capture

Training (A3):
  existing model forward -> existing RL loss -> existing backward/optimizer
  -> validated cache invalidation (M4) -> complete policy snapshot (A2/W1/R2)

Resume (M1 -> M2):
  get_model -> mto.restore_from_modelopt_state
  -> FSDP wrapping/optimizer creation -> existing distributed load_checkpoint

Save (M3):
  existing distributed checkpoint + mto.modelopt_state sidecar
  -> atomically publish one complete checkpoint
```

`mto.restore` restores structure and tensor values from `mto.save`.
`restore_from_modelopt_state` restores only structure; weights and quantizer
scales still come from RLinf's checkpoint. No HF Trainer or Accelerate launcher
is introduced inside Ray workers. Core worker modules do not import ModelOpt.

## Proposed configuration interface (not enabled)

```yaml
actor:
  precision:
    prepared_checkpoint: /path/to/prepared_policy.pt
    resume_checkpoint_dir: null
    recipe_path: /path/to/openpi_fp8_safe_v1.yaml
    real_forward: false
rollout:
  precision:
    prepared_checkpoint: /path/to/prepared_policy.pt
    real_forward: false
```

Preparation consumes an equivalent `precision` section. A future Hydra preset
would provide these settings. Recipe files use ModelOpt's schema and select
eligible FP8/NVFP4 layers. Execution precision remains separate from existing
model storage and FSDP communication precision. Conflicting legacy `ckpt_path`
and precision checkpoint settings must be rejected explicitly.

## Boundaries and unresolved implementation work

- **Arithmetic:** Standard QAT simulates quantization. The surveyed real FP8/
  NVFP4 GEMM backends use higher-precision backward operations. The proposed
  `CompressConfig(compress={"default": False}, quant_gemm=True)` keeps trainable
  weights unpacked. Packed weights are unsuitable for full-parameter updates.
- **Policy identity:** Weights, recipe hash, scale revision and execution mode
  form one behavior-policy version. Persistent buffers are already collected,
  but metadata/cache handling is additional work. Preserve the probability of
  the actual behavior policy in PPO; BF16 recomputation is not an automatic fix.
- **Caching:** No certified public cache-refresh API is selected. Require an
  upstream-supported contract before enabling real-forward updates or graphs.
- **Numerics:** Preserve FP32 log densities, ratios, advantages/returns and
  reductions. Begin with eligible large linear/MLP layers; keep sensitive action,
  value and perception paths higher precision. Calibrate using robot data.
- **Compatibility:** This fork's newer main already has a separate embodied
  actor file, native PyTorch OpenPI and a PyTorch 2.11 optional environment.
  The older survey's PyTorch 2.6 and patched-OpenPI assumptions do not describe
  this base. Recheck actual installed dependencies; relevant Transformers
  constraints still differ from ModelOpt main's `[hf]` extra. Do not enable TE
  low-precision autocast on the same ModelOpt-quantized modules.
- **Rollout lifetime:** Start synchronously without graph capture/offload.
  Async activation, compiled graphs, offload and different actor/rollout recipes
  require their own tests. Do not silently apply a partially updated snapshot.

## Implementation gates (future work)

1. Lock a compatible OpenPI/ModelOpt environment and verify simulated FP8.
2. Validate pre-wrap conversion, optimizer updates, two-GPU FSDP2, scale sync,
   unchanged-policy PPO ratios and checkpoint/resume equivalence.
3. Define the cache/snapshot contract and validate real FP8 forward dispatch.
4. Add NVFP4 and callback-driven mixed-precision search on supported Blackwell
   hardware, then measure task quality and complete RL iteration performance.

No GPU performance or convergence claim is made by this draft. Validation of
this proposal consists of Python syntax checks, AST equality against the base
for annotated files, empty ASTs for the two sketches and diff whitespace checks.
