# Greenfield-план

## Цель

Репозиторий измеряет один практический путь: RGB JPEG на локальном диске VM
становится синхронизированным batch на CUDA, готовым для модели. Это помогает
выбрать реализацию аугментаций; benchmark не измеряет скорость или качество
обучения модели.

Сейчас runnable только RGB. 9-channel, video и volume — такие же будущие
families, но до утверждения собственных input format, recipes, dataset,
batch size и output contract в репозитории для них нет runnable кода, данных
или результатов.

[Контракт выполнения](benchmark_execution_contract.md) задаёт нормативные
детали. Этот файл объясняет устройство проекта и условия готовности следующей
family.

## Текущий RGB-прогон

Одна production-клетка — это `(family, implementation, recipe, seed)`. Она
одним проходом измеряет throughput и peak process GPU memory, проверяет output
`CUDA float16 B×3×224×224` и сразу публикует immutable JSON в GCS. Следующий
запуск валидирует уже опубликованные клетки и считает только отсутствующие.

Зафиксированный workload:

- первые 10 000 JPEG из лексикографически отсортированного ImageNet validation;
- 57 вручную сопоставленных recipes и семь implementations;
- batch 256, один warm-up batch и 32 измеряемых batch;
- seeds 137, 138 и 139;
- одна Standard `g2-standard-16` VM с NVIDIA L4;
- 15 persistent DataLoader workers, `pin_memory=True`, `prefetch_factor=2`.

Одна VM один раз восстанавливает либо собирает locked RGB environment,
скачивает code archive и dataset. Затем она prewarm-читает выбранные файлы вне
таймера и последовательно проходит недостающие клетки. Job — это одна library
implementation со всеми её recipes и seeds, а не shard по времени, recipe или
seed.

## Путь данных и измерение

```text
JPEG на диске
  → reader и decoder выбранной библиотеки
  → recipe
  → collate
  → pinned H2D
  → оставшийся GPU recipe
  → GPU Normalize
  → проверка output и CUDA synchronize
```

`Compose` init, import и preflight не входят в throughput. Monitor NVML
запускается до построения pipeline, поэтому peak GPU memory включает
construction, reader, decoder, workers, buffers и CUDA allocations.

Normalize никогда не выполняется на CPU. AlbumentationsX, Pillow и TorchVision
CPU копируют `uint8`; Kornia CPU применяет свой recipe в `float32`, затем
готовый host batch преобразуется в `float16` перед H2D. GPU-tail TorchVision и
Kornia выполняют только минимальный CPU-prefix, который делает samples
collatable; остальной recipe и Normalize идут на GPU в `float16`. DALI использует
свой native file reader, mixed decoder и GPU graph.

## Маленькая архитектура

```text
family config (archive + selection) + cloud config + recipe catalog + environment lock
                                      |
                                      v
                           immutable RunRecord и matrix
                                      |
                                      v
                            одна resumable GCP L4 VM
                                      |
                                      v
                            immutable GCS cell records
                                      |
                                      v
                     aggregation и paper только из complete rows
```

`run_config.py` описывает параметры family. `frozen_rgb_run.py` хеширует
входы в run record. `gcp_controller.py` создаёт или возобновляет ровно одну
помеченную VM. `guest_worker.py` запускает недостающие клетки. `rgb_executor.py`
задаёт timed path и output validation. Adapters содержат только
library-specific reader, decoder и recipe pipeline.

## Инварианты

1. Изменение recipes, archive identity или selection rule, package lock, code archive, hardware или
   timing boundary создаёт новый run record.
2. Throughput и GPU memory принадлежат одной клетке; отдельного memory run нет.
3. Нет microbenchmarks, isolated H2D, training loops, capacity sweeps и
   decoder-only production matrices: это другие вопросы.
4. Каждая успешная клетка немедленно публикуется в GCS. Повторный запуск
   читает и валидирует её, а не считает заново.
5. Сравнение скорости и memory идёт только по общему supported recipe
   intersection. Coverage — отдельный catalog census.
6. Если клетка отсутствует или невалидна, исправляют execution path и
   генерируют данные; статья не подменяет их объясняющим текстом.

## Добавление следующей family

Для 9ch, video или volume сначала подготавливают dataset archive с одним
документированным sample format. Затем добавляют:

1. family config с output shape, dtype, batch size, единицей throughput,
   seeds и timing window;
2. компактное правило отбора данных, recipe catalog и support matrix;
3. implementations с GPU-only Normalize и output validation;
4. один L4 preflight batch для каждой пары implementation–recipe; non-DALI path использует один временный DataLoader worker, DALI — native graph;
5. новый immutable run и полный matrix.

RGB batch size, sources, recipes и results не переносятся в новую family по
умолчанию.
