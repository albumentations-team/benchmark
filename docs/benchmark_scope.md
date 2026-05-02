# Benchmark Scope

## Library Sets

RGB image benchmarks compare four libraries:

- `albumentationsx`
- `torchvision`
- `kornia`
- `pillow`

9-channel image benchmarks compare three libraries:

- `albumentationsx`
- `torchvision`
- `kornia`

Pillow is excluded from 9-channel benchmarks because its direct image API is RGB/PIL-image oriented.

## Transform Selection

The shared transform catalog lives in `benchmark/transforms/specs.py`.

For RGB benchmarks, a transform is eligible when it exists in at least two of the four RGB libraries.
For 9-channel benchmarks, a transform is eligible when it exists in at least two of the three 9-channel libraries.
For video benchmarks, a transform is eligible when it exists in at least two selected video libraries.

The paper transform sets are fixed in:

- `docs/paper_transform_sets/rgb.md`
- `docs/paper_transform_sets/9ch.md`
- `docs/paper_transform_sets/video.md`

Use them from the CLI with:

```bash
--transform-set paper
```

Each library still reports only transforms it supports directly. We do not recreate missing library features with large benchmark-side helper implementations just to force every library to have every row.

## DataLoader Pipeline Recipes

DataLoader pipeline benchmarks measure training-style recipes, not isolated transforms.
Micro benchmarks are different: they measure only the named transform in the library's native format and must not add
`Normalize`, `ToTensor`, axis conversion, or DataLoader collation work.

For non-crop transforms, the recipe shape is:

```text
RandomCrop224 + <transform> + Normalize + ToTensor
```

For crop transforms, the crop replaces `RandomCrop224`:

```text
<Crop> + Normalize + ToTensor
```

Example recipe names:

```text
RandomCrop224+Affine+Normalize+ToTensor
RandomCrop224+Brightness+Normalize+ToTensor
RandomResizedCrop+Normalize+ToTensor
CenterCrop224+Normalize+ToTensor
```

`Normalize` is not benchmarked as a pipeline augmentation because it is already part of every pipeline recipe.

`ToTensor` is implemented in the pipeline spec, not in the generic runner. AlbumentationsX pipeline recipes use
`Normalize` followed by `ToTensorV2`; Pillow pipeline recipes use `torchvision.transforms.PILToTensor` before normalization;
torchvision and Kornia already operate on tensors. The DataLoader runner should receive fixed-shape recipe outputs that
PyTorch can default-collate into a single batched tensor, and should not guess or repair channel layout. The
`decode_dataloader_augment_batch_copy` scope adds batch tensor materialization and optional CUDA/MPS transfer; default
collation itself is part of every DataLoader scope.

## Slow Transform Guard

Micro and DataLoader pipeline benchmarks run a preflight check before spending the full benchmark budget on a transform.

For image benchmarks, transforms slower than `0.05` seconds per image are early-stopped by default. This corresponds to `20 img/s`. For video benchmarks, transforms slower than `2.0` seconds per video are early-stopped by default.

The result JSON keeps an `early_stopped` entry with the preflight throughput and reason, instead of hanging the benchmark on transforms that are too slow for practical training use.

This guard is especially important for DataLoader benchmarks. A full pipeline run can multiply by:

```text
number of recipes * number of runs * full dataset epochs
```

Very slow transforms can otherwise make the benchmark appear stuck and block the paper sweep. The early-stop policy is intentional: transforms below the practical throughput floor are not useful DataLoader candidates.

Use `--disable-slow-skip` only when explicitly measuring slow transforms.

## Architecture Source Of Truth

Benchmark policy is intentionally centralized:

- `benchmark/matrix.py` declares scenario/mode/library support, spec paths, requirement groups, paper transform-set files,
  device policy, pipeline scopes, and backend names.
- `benchmark/policy.py` declares media defaults and slow-transform preflight thresholds.
- `benchmark/jobs.py` builds immutable benchmark jobs and subprocess commands.
- `benchmark/orchestrator.py` dispatches jobs to pyperf, DataLoader, or DALI pipeline backends.
- `benchmark/envs.py` owns joined virtualenvs, dependency refresh, and dependency cache keys.
- `benchmark/specs/load.py` and `benchmark/media/loaders.py` keep spec validation and media loading out of the CLI.

Do not add new benchmark matrix constants directly to `benchmark/cli.py`. Add them to `benchmark/matrix.py`, then extend
tests in `tests/test_matrix.py` and `tests/test_jobs_orchestrator.py`.

For a full module map, see `docs/benchmark_architecture.md`.

## Visual Progress

Long benchmark runs must show tqdm progress with descriptive labels. Progress bars should make it clear which dimension is moving:

- Library loops: `<scenario>/<mode>`.
- Media loading: `Load images (<library>, <channels>ch)` or `Load videos (<library>)`.
- Micro transforms: `Micro transforms (<library>, <media>)`.
- Pyperf micro transforms: `Pyperf micro transforms (<library>, <media>)`.
- Pipeline transforms: `Pipeline transforms (<library>, <scope>, w=<workers>, b=<batch_size>)`.

Do not add anonymous tqdm bars. Every tqdm must have a useful `desc` and a unit such as `lib`, `img`, `video`, or `transform`.

## Paper Run Plan

The paper does not need the full benchmark matrix on every CPU vendor. Run the complete CPU suite once on a modern Intel VM, run a small AMD sanity check, and run video GPU benchmarks separately.

### Main CPU Suite

Machine: `c4-standard-16` or equivalent modern Intel CPU.

Run these as the main paper tables:

- RGB micro benchmark: `image-rgb`, `micro`, libraries `albumentationsx torchvision kornia pillow`, transforms from `docs/paper_transform_sets/rgb.md`.
- 9-channel micro benchmark: `image-9ch`, `micro`, libraries `albumentationsx torchvision kornia`, transforms from `docs/paper_transform_sets/9ch.md`.
- RGB DataLoader memory pipeline: `image-rgb`, `pipeline`, `memory_dataloader_augment`.
- RGB DataLoader disk/decode pipeline: `image-rgb`, `pipeline`, `decode_dataloader_augment`.
- 9-channel DataLoader memory pipeline: `image-9ch`, `pipeline`, `memory_dataloader_augment`.
- 9-channel DataLoader disk/decode pipeline: `image-9ch`, `pipeline`, `decode_dataloader_augment`.
- Video benchmarks use transforms from `docs/paper_transform_sets/video.md`.

Recommended DataLoader settings for final paper runs:

```text
--batch-size 256
--workers 8
--num-runs 3
--min-time 0
--thread-policy pipeline-single-worker
```

Use the full ImageNet validation set for final DataLoader runs. For cheaper iteration, keep the same production path and reduce only explicit sizing flags, for example `--num-items 1000 --batch-size 64 --workers 8 --num-runs 1 --min-time 0`.

### AMD Sanity Check

Machine: `c4d-standard-16` or equivalent modern AMD CPU.

Run these only to confirm trends do not invert on another modern CPU:

- RGB micro benchmark: `image-rgb`, `micro`, libraries `albumentationsx torchvision kornia pillow`.
- Optional reduced RGB DataLoader sanity run: `image-rgb`, `pipeline`, one of `memory_dataloader_augment` or `decode_dataloader_augment`, with `--num-items 1000` or another small subset.

Do not rerun the full RGB + 9ch + DataLoader matrix on AMD unless the paper explicitly studies CPU-vendor effects.

### GPU / Video Suite

Machine: `g2-standard-16` with an L4 GPU, or equivalent.

Run these for video/GPU tables:

- GPU video micro benchmarks for GPU-capable libraries, especially `torchvision` and `kornia`.
- GPU video DataLoader/pipeline benchmarks for GPU-capable paths.
- DALI video pipeline benchmarks when DALI is available on the target image.

CPU-only image rows should not be rerun on GPU machines for hardware symmetry. Label each result row with the machine class that actually ran it.

### Validation

After pulling artifacts, validate coverage before producing tables:

```bash
python -m tools.check_paper_coverage gcp_runs output
```

Use optional-library coverage checks only when DALI is required for the current table.
