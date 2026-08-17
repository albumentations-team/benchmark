# Контракт production-бенчмарка

Репозиторий измеряет путь от подготовленного файла на локальном диске до batch, готового к входу модели на GPU. Сейчас запускается только RGB. 9-channel, video и volume используют тот же runner и схему результата, но появятся после утверждения собственных данных, recipes, output contract и batch size.

## Что измеряет одна клетка

Клетка имеет идентичность `(family, implementation, recipe, seed)`. Успешная RGB-клетка одним проходом записывает:

- throughput от JPEG на диске до готового batch;
- peak process GPU memory от построения pipeline до последнего CUDA synchronize;
- проверенный output: CUDA `float16`, BCHW, `B×3×224×224`;
- версии библиотеки и GPU как краткую диагностику.

Отдельный memory run запрещён. CPU memory, microbenchmark, isolated H2D, decoder-only benchmark, training loop и maximum-batch sweep не отвечают на этот вопрос и не входят в production.

## Зафиксированный RGB workload

| Поле | Значение |
| --- | --- |
| Данные | первые 10 000 `val/*.JPEG` из лексикографически отсортированного SHA-256-проверенного ImageNet validation archive |
| Recipes | 57 вручную сопоставленных RGB recipes; каждая доступна хотя бы в двух библиотеках |
| Matrix | 253 supported library–recipe pairs × 3 seeds = 759 клеток |
| Seeds | 137, 138, 139 |
| Batch | 256 изображений |
| Warm-up | 1 полный batch вне таймера |
| Измерение | 32 полных batch, 8 192 изображения |
| Workers | 15, `persistent_workers=True`, `prefetch_factor=2`, `pin_memory=True` |
| Hardware | одна Standard `g2-standard-16` с одной NVIDIA L4 в любой подходящей GCP zone |

`prefetch_factor=2` фиксирован для всех PyTorch DataLoader paths. Он даёт каждому worker до двух подготовленных batch и не становится параметром, который подбирают под библиотеку.

## Timing boundary

```text
JPEG на локальном диске
  → reader и decoder выбранной библиотеки
  → recipe
  → collate
  → pinned H2D
  → оставшийся GPU recipe
  → GPU Normalize
  → output validation
  → CUDA synchronize
```

`Compose` init, import и preflight не входят в throughput. NVML monitor стартует до построения pipeline, поэтому peak memory включает reader, decoder, workers, buffers и GPU allocations. Он работает в том же pass, что throughput.

Normalize никогда не выполняется на CPU. Все output rows заканчиваются CUDA `float16` Normalize:

| Implementation | До GPU | На GPU |
| --- | --- | --- |
| AlbumentationsX CPU | SimpleJPEG, `uint8` recipe и `uint8` pinned H2D | convert в `float16`, Normalize |
| Pillow CPU | Pillow decode, `uint8` recipe и `uint8` pinned H2D | convert в `float16`, Normalize |
| TorchVision CPU | `torchvision.io`, `uint8` recipe и `uint8` pinned H2D | convert в `float16`, Normalize |
| Kornia CPU | `torchvision.io`, CPU recipe в `float32`, host batch cast в `float16`, pinned H2D | Normalize в `float16` |
| TorchVision GPU | `torchvision.io` и минимальный CPU-prefix до collatable shape | tail и Normalize в `float16` |
| Kornia GPU | `torchvision.io` и минимальный CPU-prefix до collatable shape в `float32`, host batch cast в `float16` | tail и Normalize в `float16` |
| DALI GPU | native file reader и mixed JPEG decoder | native DALI graph и `float16` Normalize |

JPEG имеют разные исходные размеры. Для каждого non-DALI path CPU выполняет ровно минимальный prefix recipe до первого шага, который даёт collatable `224×224` samples; остальной recipe выполняется на batch там, где implementation это поддерживает. `Resize` означает ровно `Resize(224) → Normalize → ToTensor`: hidden crop не добавляется. Если crop нужен после `Pad`, `LongestMaxSize` или `SmallestMaxSize`, он записан явным stage recipe.

Для одного operation parameters сопоставляются вручную по практическому эффекту. Pixel-exact равенство не требуется: APIs, interpolation, RNG и rounding различаются.

## Доступ к данным

После распаковки VM один раз последовательно читает все 10 000 JPEG вне timed cells. Затем для каждого seed все implementations используют одну детерминированную PCG64-перестановку без повторов. PyTorch DataLoader не делает дополнительный shuffle; DALI получает тот же уже переставленный file list.

Это измеряет reader, decode и augmentation через filesystem API, но не physical-disk bandwidth: байты после prewarm обычно приходят из Linux page cache. Такое состояние убирает случайное преимущество первой библиотеки с cold cache и сохраняет training-like random access order.

## Run, VM и resume

`augbench launch-rgb` разрешён только из чистого Git worktree. Он создаёт source archive текущего commit, собирает immutable `run.json` из family config (включая SHA-256 архива и правило отбора), package lock, recipe catalog и hardware config, а затем:

1. проверяет GCS `runs/<run_id>/cells/*.json`;
2. валидирует каждую найденную клетку по содержимому и identity;
3. не создаёт VM, если существует активная VM с тем же `augbench-run` label;
4. иначе ищет все zones с `g2-standard-16` и L4, затем создаёт одну Standard VM;
5. VM скачивает code, raw dataset и cached RGB environment один раз, извлекает первые 10 000 отсортированных `val/*.JPEG`, prewarms их, выполняет preflight и идёт по отсутствующим клеткам;
6. после каждой клетки сразу создаётся immutable GCS object;
7. после terminal VM следующий `launch-rgb` удаляет именно эту маркированную terminal VM и продолжает с отсутствующих клеток.

В VM один implementation job последовательно проходит все его recipes и seeds. Нет job на recipe, seed, duration или shard.

## Aggregation

Агрегация начинается только после полного validated matrix. Speed и GPU memory сравниваются построчно для одинаковых family, recipe, seed policy, batch, dataset, hardware и boundary. Pairwise и all-library summaries используют точное пересечение supported recipes. Coverage census хранится отдельно: большее число operations не превращается в unsupported speed loss.

Если отсутствует нужная метрика или результат, исправляют execution path и получают данные. Текст статьи не маскирует такой пробел объяснением.

## Добавление следующей family

Перед первой 9ch/video/volume production matrix нужно добавить:

1. подготовленный dataset archive и компактное детерминированное правило отбора с одним стандартизированным sample format;
2. family config с output shape, dtype, batch size, seeds и units throughput;
3. recipe catalog и support matrix;
4. implementations с GPU-only Normalize и output validation;
5. one-batch preflight на L4;
6. новый run и complete matrix.

RGB recipes, batch size, data format и результаты не переносятся в новую family по умолчанию.
