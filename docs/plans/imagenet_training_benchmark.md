# ImageNet training-step benchmark: implementation plan

Status: Phases 0, 1, and 2 are implemented and pass the local tests. The L4 preflight has not run yet.
Task 2.1 added no `operations.yaml` entry: that census describes AX's public API and nothing reads it for recipes.

## Goal

Add a second workload next to the RGB input-pipeline benchmark. It runs an ImageNet-style training
step (forward, backward, optimizer step) on batches delivered by the same eight input paths.
The question it answers: does the loader ranking survive when the model shares the GPU and the
CPU cores? Done means:

- `uv run augbench launch --config configs/families/train-resnet18.yaml` and the `train-resnet50.yaml`
  twin each produce a complete immutable run on the L4 fleet.
- Every cell records training images/s, the model-only ceiling measured in the same process,
  loader wait time, and two GPU-memory peaks.
- Existing RGB and grayscale runs, cell IDs, run IDs, and generators behave exactly as before.

## Constraints

Decisions from the design discussion, unchanged:

- Data: all 50,000 ImageNet validation JPEGs (`first-sorted`). Per seed, one pass without rollover.
- Models: ResNet-18 and ResNet-50 from `torchvision.models` with `weights=None`. Two models are two
  runs with their own config files. `CellKey` does not change.
- Step: `torch.compile` (default mode), bfloat16 autocast, no `GradScaler`, `channels_last`, SGD with
  momentum, cross-entropy on fixed random labels. Eager optimizer.
- Recipes. Stage definitions for the first three are copied verbatim from `catalog/recipes/rgb.yaml`.
  Pillow paths lack `ColorJitter` and `Erasing`; those recipes run on six paths.
  1. `RandomResizedCrop+HorizontalFlip` (8 paths)
  2. `...+ColorJitter` (6 paths)
  3. `...+ColorJitter+Erasing` (6 paths)
  4. `RandomResizedCrop+HorizontalFlip+RandAugment` (8 paths). A fifth recipe with Erasing was dropped by the owner.
- RandAugment: use the library's native policy where one exists (TorchVision v2 `RandAugment`, Kornia
  `kornia.augmentation.auto.RandAugment`, DALI `nvidia.dali.auto_aug.rand_augment`). Assemble it where none
  exists (AlbumentationsX, Pillow, Pillow-AllSIMD). Shared parameters: 2 operations, magnitude 9 of 31 bins
  (Kornia: `n=2, m=9`), fill 0. Each image draws its own operations.
- The loader output contract is unchanged: CUDA `float16`, BCHW, normalized on the GPU.
- Interpolation: bilinear everywhere, without antialiasing, as in the existing contract. `RandomResizedCrop`
  keeps `antialias: false` from the RGB catalog. The `RandAugment` warps (shear, translate, rotate) are one
  bilinear warp each and have no antialiasing pass on any path.
- Metrics in one pass: images/s over the window, ceiling images/s, loader wait seconds, peak GPU
  memory over the whole cell (contract) and peak after warm-up.
- Workers 15, persistent, pin memory, prefetch factor 2. Seeds 137–141. Same machine, same lock.
- Old RGB results stay valid: no change to `CellKey`, `RunRecord` identity inputs, or stored bytes.

## Out of scope

- TrivialAugment, AutoAugment, MixUp, CutMix, label smoothing, EMA, accuracy, convergence.
- `ColorJitter` and `Erasing` for Pillow and Pillow-AllSIMD.
- README, paper, and report-generator changes for the new workload.
- `GradScaler`, mixed model dtype experiments, `torch.compile` modes beyond the default.

## Tasks

### Phase 0: neutral names, no behavior change

**Task 0.1: pin RGB identities (red-green guard)**
- Modify: `tests/test_run_records.py`.
- Add a test with a literal `cell_id` for one fixed `CellKey`, and one literal `run_id` for a fixed
  `RunRecord` built with `build_run_record`. Take the values by running the current code once.
- Run `uv run pytest tests/test_run_records.py` (green before and after Phase 0).

**Task 0.2: rename generic machinery**
- Rename: `frozen_rgb_run.py` → `frozen_run.py` (`FrozenRgbRun` → `FrozenRun`, `build_frozen_rgb_run` →
  `build_frozen_run`), `launch_rgb.py` → `launch.py` (`launch_rgb` → `launch`), `rgb_executor.py` →
  `cell_executor.py` (`RGBCellExecutor` → `CellExecutor`). Use `git mv`.
- Modify: `cli.py`, `guest_worker.py`, `tests/test_frozen_rgb_run.py` → `tests/test_frozen_run.py`,
  `tests/test_cli.py`, `tests/test_guest_worker.py`, `tests/test_active_module_dependencies.py`,
  and every doc that links the old file names (`docs/greenfield_plan.md`, `docs/benchmark_execution_contract.md`).
- Verify with `grep -rn "frozen_rgb_run\|launch_rgb\|rgb_executor\|RGBCellExecutor" src tests docs`
  returning nothing, then `uv run pytest`.

**Task 0.3: one launch command and a grayscale YAML**
- Create: `configs/families/grayscale.yaml` (the config `grayscale_config()` builds today, with the
  archive URI and SHA-256 as placeholders that the command line overrides).
- Modify: `cli.py`. Add `augbench launch --config PATH`. Keep `launch-rgb` (config
  `configs/families/rgb.yaml`) and `launch-grayscale` (loads `grayscale.yaml`, overrides the archive
  fields from its arguments) as thin aliases. Delete `grayscale_config()` from `launch.py`.
- Test: `tests/test_cli.py` calls each command with a monkeypatched `launch` and checks the config
  path; `tests/test_grayscale.py::test_grayscale_colorize_has_a_separate_frozen_matrix` keeps its
  30-cell assertion.

### Phase 1: the training family

**Task 1.1: config and identity**
- Modify: `run_config.py`, `run_records.py`, `recipes/models.py`, `frozen_run.py`.
- Add `TrainingConfig` (frozen pydantic): `model: Literal["resnet18", "resnet50"]`, `compile: bool = True`,
  `amp_dtype: Literal["bfloat16"] = "bfloat16"`, `learning_rate: float`, `momentum: float`,
  `weight_decay: float`, `ceiling_warmup_steps: int`, `ceiling_steps: int`, `label_classes: int = 1000`.
- `FamilyRunConfig.training: TrainingConfig | None = None`. Add `"imagenet_train"` to the family literal,
  to `CellKey.family`, and to `RecipeCatalog.family`. Validation: the family `imagenet_train` requires
  `training`, every other family forbids it.
- `frozen_run.py` dumps the family config with `exclude={"training"}` when it is `None`, so RGB
  `family_config` bytes stay identical.
- Test (red first): `FamilyRunConfig` for `configs/families/rgb.yaml` dumps no `training` key; the
  pinned RGB `run_id` from Task 0.1 is unchanged; a config with `family: imagenet_train` and no
  `training` raises `ValueError`.

**Task 1.2: recipes and configs**
- Create: `catalog/recipes/imagenet_train.yaml` (family `imagenet_train`, recipes 1–3; Phase 2 adds 4 and 5),
  `configs/families/train-resnet18.yaml` (batch 256), `configs/families/train-resnet50.yaml` (batch 128, to
  leave VRAM for DALI's 1.7 GiB and compile workspace). Both: item_count 50000, warm-up 10 batches,
  150 measured batches, 15 workers, prefetch 2, seeds 137–141, `cloud: configs/cloud/gcp-l4.yaml`,
  the eight implementations.
- Modify: `recipe_contract.py` only if `validate_rgb_recipes` rejects the new family name; the
  stage rules stay.
- Test: `tests/test_matrix.py` style test builds the matrix for each config: 100 cells per run
  (5 seeds × (8 + 6 + 6)) after Phase 1 and 140 after Phase 2 (5 seeds × (8 + 6 + 6 + 8)); `required_items` (256 × 160 = 40,960; 128 × 160 = 20,480) does not
  exceed 50,000.

**Task 1.3: measurement hooks**
- Modify: `measurement_window.py`, `nvml_memory.py`, `run_records.py`.
- `MeasurementWindow.waits: list[float] | None` (a field of the window, because `measure_window` is at the
  argument limit) receives, for each measured batch, the seconds spent inside `next(batches)`; warm-up batches
  are not recorded.
- `NvmlProcessMemoryMonitor.mark_steady()` stores the peak seen so far as a boundary and a later
  `steady_peak_mib()` returns the maximum over samples taken after the mark.
- `TrainingObservation` (frozen): `model`, `ceiling_images_per_second`, `loader_wait_seconds`,
  `steady_peak_mib`, `loss_finite: bool`. `ResultRecord.training: TrainingObservation | None = None`.
- Test (red first): fake clock shows `waits` sums only measured batches; fake sampler shows the
  steady peak ignores a spike recorded before `mark_steady()`; `encode_result` of an RGB record contains
  no `"training"` key (the existing `exclude_none=True` is the mechanism).

**Task 1.4: model and step**
- Create: `src/augbench/training/__init__.py`, `training/model.py`, `training/step.py`.
- `build_model(name: str) -> torch.nn.Module`: `torchvision.models.resnet18/resnet50(weights=None)`,
  `.cuda().to(memory_format=torch.channels_last)`.
- `TrainSession(config: TrainingConfig, batch_size: int)`: compiles the model once per process and
  keeps an initial `state_dict`. `reset(seed)` reloads the initial weights in place (no recompile),
  creates a fresh `torch.optim.SGD(..., fused=True)`, and a fixed `torch.randint` label tensor from
  the seed.
- `TrainSession.step(batch) -> int`: `autocast(device_type="cuda", dtype=torch.bfloat16)`, forward,
  `cross_entropy`, `backward`, `optimizer.step()`, `zero_grad(set_to_none=True)`; returns
  `batch.shape[0]`. `TrainSession.loss_is_finite() -> bool` reads the last loss outside the timer.
- `TrainSession.ceiling(images: Tensor, steps: int, warmup: int) -> float`: images/s on one synthetic
  `float16` BCHW batch already on the GPU; the warm-up includes compilation.
- Test: CPU-only unit test with a tiny fake model checks that `reset` restores weights, `step` returns
  the batch size, and labels depend only on the seed. A `@pytest.mark.skipif(no CUDA)` test runs one
  real ResNet-18 step.

**Task 1.5: executor integration**
- Modify: `cell_executor.py`.
- When `config.training` is set, `CellExecutor` owns one `TrainSession`. In `execute`: build the
  recipe, run the ceiling (`ceiling_warmup_steps` + `ceiling_steps`) before the loader opens, call
  `monitor.mark_steady()`, then run `measure_window` with `consume` that calls
  `ReadyGpuBatchConsumer.prepare(batch)` and `session.step(ready)`; the DALI path wraps its existing
  `consume` the same way. After the window, `loss_finite` and `waits` fill `TrainingObservation`.
- `preflight` runs one real training step on the one-worker loader (compile happens once per process).
- Test: with a fake loader and a fake session, `execute` returns a `ResultRecord` whose `training`
  holds the ceiling, the wait sum, and `loss_finite`; for an RGB config it returns `training is None`.

**Task 1.6: VM environment**
- Modify: `infra/gcp/bootstrap.sh`, `guest_worker.py`.
- Export `TORCHINDUCTOR_CACHE_DIR=/var/lib/augbench/inductor` for every worker (same cache on one VM
  for later jobs) and fail early with a clear message if `gcc` is missing when `training` is set.
- Test: `tests/test_gcp_bootstrap.py` asserts the variable and the `gcc` check appear for a request
  whose family config has `training`.

**Task 1.7: documents**
- Modify: `docs/benchmark_execution_contract.md` (new section "ImageNet training workload": step,
  window, metrics, two peaks, ceiling, recipes, why two runs), `.codex/skills/augmentation-benchmark-contract/SKILL.md`
  (record that this research question was added explicitly and where it differs), `environments/README.md`
  (one lock serves all image families), `docs/greenfield_plan.md`.

### Phase 2: RandAugment (native where it exists, assembled elsewhere)

Phase 1 ships on its own. Phase 2 adds recipe 4 and one new operation, `RandAugment`.

**Task 2.1: operation, parameters, contract rule**
- Modify: `catalog/operations.yaml` (new operation `RandAugment`, domain `policy`, so the coverage census
  counts stay as they are; support: TorchVision, Kornia, DALI `direct`, AlbumentationsX and Pillow `composed`),
  `docs/benchmark_execution_contract.md`.
- Stage parameters in the recipe: `num_ops: 2`, `magnitude: 9`, `num_magnitude_bins: 31`,
  `interpolation: bilinear` (the repository convention for every warp), `fill: 0`, and `operations` listing the 14 operations of
  TorchVision's pool: `Identity, ShearX, ShearY, TranslateX, TranslateY, Rotate, Brightness, Color, Contrast,
  Sharpness, Posterize, Solarize, AutoContrast, Equalize`.
- Contract rule (extension approved by the owner): the AX adapter may use `A.SomeOf` and `A.OneOf` of stock
  transforms inside the `RandAugment` stage only. Everywhere else the existing ban on `A.Lambda`, `A.OneOf`, `A.Sequential` stays.
- Test: the catalog loads; `tests/test_image_recipe_specs.py::test_ax_recipe_stages_are_stock_transforms`
  accepts `SomeOf` for the `RandAugment` stage and still rejects `Lambda` everywhere.

**Task 2.2: magnitude table and assembled paths**
- Create: `src/augbench/implementations/randaugment.py` with the constants every assembled path uses,
  taken from TorchVision v2 at magnitude 9 of 31 bins (verified on 224×224):
  shear 0.09; translate 30.45 px (0.136 of the side); rotate 9°; brightness, color, contrast, sharpness factor
  1 ± 0.27; posterize 7 bits; solarize threshold 0.7 of the maximum; signed operations pick their sign
  uniformly for each application; each of the two draws picks one of the 14 operations uniformly, with
  replacement.
- Modify: `implementations/albumentationsx_impl.py`: stage = `A.SomeOf(14 stock transforms, n=2, replace=True)`;
  a signed operation is `A.OneOf([op(+m), op(-m)])` so magnitude stays fixed; `Identity` is `A.NoOp`.
  Write each operation's stock mapping into a table in the module. Sharpness is `UnsharpMask` with a signed alpha (AX #699, released in 2.6.9).
- Modify: `implementations/pillow_impl.py` (shared by Pillow-AllSIMD): same table with `ImageOps` and
  `ImageEnhance` plus `Image.transform(AFFINE)`; operations and signs are drawn per image in the adapter.
- Test (red first): a unit test pins the constants against `torchvision.transforms.v2.RandAugment` (skipped
  without torchvision); one test per operation checks the expected pixel effect at the fixed magnitude on
  a small image; two images in one worker call draw different operations under different RNG state.
- Warp operations (shear, translate, rotate) are bilinear on every path: AX `Affine(interpolation=cv2.INTER_LINEAR)`,
  Pillow `Image.transform(AFFINE, BILINEAR)`. Kornia's `RandAugment` warps already use `Resample.BILINEAR`
  (checked on its rotation, shear, and translation modules), so it needs no argument. A test asserts it per path.

**Task 2.3: native TorchVision**
- Modify: `implementations/torchvision_impl.py`, `torchvision_pipeline_impl.py`:
  `tv_transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=BILINEAR, fill=0)`
  on one uint8 image per call. TorchVision GPU applies it per image on CUDA tensors, as it does for other stages.
- Test: the builder returns a native `RandAugment` with those arguments.

**Task 2.4: native Kornia**
- Modify: `implementations/kornia_impl.py`, `kornia_pipeline_impl.py`:
  `kornia.augmentation.auto.RandAugment(n=2, m=9)`. Kornia draws one operation sequence per call and
  samples without replacement from its own pool, so call it once per image. On the CPU path the worker
  already calls it per sample. On the GPU path loop over the batch. Record both facts as differences in
  `docs/torchvision_kornia_adapters.md`.
- Test: two images in one batch can receive different operation sequences on the GPU path (CPU-only test
  with a stub policy that records calls).

**Task 2.5: native DALI**
- Verified in the DALI v2.4.0 sources (`nvidia/dali/auto_aug/rand_augment.py`, `augmentations.py`,
  `core/_select.py`): `rand_augment(data, n, m, num_magnitude_bins=31, shape=None, fill_value=128,
  interp_type=None, max_translate_abs=None, max_translate_rel=None, seed=None, monotonic_mag=True, excluded=None)`.
  It takes uint8 `HWC` samples, draws `n` operation indices per sample with `fn.random.uniform` (independent
  draws, so with replacement), negates signed magnitudes per sample, and runs the chosen operation by
  splitting the batch through a binary tree of conditionals (needs `enable_conditionals=True`; the
  pipeline in `adapters/dali/native.py` already sets it). The pool is the same 14 operations as TorchVision with the
  same ranges (shear 0.3, rotate 30, enhance 0.9, posterize 8→4 bits, solarize 256→0). At `m=9` of 31 bins
  it gives shear 0.09, rotate 9°, enhance 1 ± 0.27, posterize 7 bits, solarize threshold 179.2.
- Differences from TorchVision: translation range is 100/224 of the side (30.0 px at 224) against
  150/331 (30.45 px); `fill_value` defaults to 128, so we set it to 0 (its default interpolation is already linear); Equalize follows OpenCV; Sharpness
  leaves a 1 px border unfiltered. Record them in the contract.
- Modify: `adapters/dali/native.py`: handler `RandAugment` calling
  `rand_augment(data, n=2, m=9, num_magnitude_bins=31, shape=(224, 224), fill_value=0, interp_type=types.INTERP_LINEAR)`,
  which matches the catalog parameters.
- Test: handler registered in `_STAGE_HANDLERS`; the graph build stays Linux-only and runs in the L4
  preflight, where one batch must come out `uint8` HWC before normalization.

**Task 2.6: recipe 4**
- Modify: `catalog/recipes/imagenet_train.yaml` (recipe 4), configs unchanged.
- Test: matrix tests expect 140 cells per run.

**Task 2.7: documents**
- Modify: `docs/benchmark_execution_contract.md` (RandAugment: native or assembled, shared parameters, the
  difference list: operation pools, replacement, interpolation, per-call scope), `docs/benchmark_faq.md`
  (one paragraph on why policies are native where they exist).

## Risks to test

1. RGB `run_id` or `cell_id` changes: Task 0.1 and Task 1.1 pin literal values.
2. Stored RGB bytes change when the optional `training` field is added: Task 1.3 asserts the
   encoded JSON has no `training` key.
3. Weight reset triggers a recompile or leaks optimizer state between cells: Task 1.4 asserts
   `reset` leaves the parameter identities unchanged and zeroes optimizer state; the L4 preflight
   logs recompile counts through `torch._dynamo.utils.counters`.
4. ResNet-50 out of memory next to DALI: batch 128 is the starting point; the L4 preflight is the
   judge. If it fails, lower the batch in `train-resnet50.yaml`; this creates a new run.
5. Missing C compiler on the VM image: Task 1.6 fails early with a message instead of during compile.
6. Loss turns non-finite in bfloat16: `loss_finite` is recorded and the cell fails when false.
7. Native RandAugment pools differ: TorchVision, DALI, and the assembled paths share the same 14 operations
   drawn with replacement; Kornia draws without replacement from its own pool. Tasks 2.2–2.5
   record each difference in the documents. The comparison is policy-to-policy at equal `n` and `m`, not
   pixel-identical output.
8. Kornia GPU loops over images to get per-image operations and may be slow. That is the cost of a
   per-image native policy, reported as measured (Task 2.4).
9. An assembled signed operation drifts from TorchVision's magnitude: Task 2.2 pins the constants against
   TorchVision at test time.

## Verification

```bash
uv run ruff check src tests paper && uv run ruff format --check src tests paper
uv run mypy src
uv run pytest -q
uv run augbench launch --config configs/families/train-resnet18.yaml   # L4 fleet, needs a clean commit
uv run augbench launch --config configs/families/train-resnet50.yaml
```

After each run, download its cells and check: 100 cells, all `status: ok`, every cell has `training`,
`ceiling_images_per_second` exceeds the cell's `throughput.value`, `steady_peak_mib` does not exceed
`gpu_memory.peak_mib`, and the guest logs contain no `Traceback`.
