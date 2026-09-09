from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from augbench.recipes.runtime import UnsupportedRecipeError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from augbench.adapters.sources import SourceRef
    from augbench.recipes.models import RecipeSpec, RecipeStage


class _Execution(Protocol):
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    seed: int
    height: int
    width: int


def supports_recipe(recipe: RecipeSpec) -> bool:
    return all(stage.operation_id in _STAGE_HANDLERS for stage in recipe.stages)


class DaliBatchSource:
    """A native DALI reader/decoder/augmentation graph yielding CUDA tensors."""

    def __init__(
        self,
        *,
        sources: Sequence[SourceRef],
        recipe: RecipeSpec,
        execution: _Execution,
    ) -> None:
        if not supports_recipe(recipe):
            unsupported = [stage.operation_id for stage in recipe.stages if stage.operation_id not in _STAGE_HANDLERS]
            raise UnsupportedRecipeError(f"DALI does not implement recipe stages {unsupported}")
        self._sources = sources
        self._recipe = recipe
        self._execution = execution
        self._pipeline: Any | None = None
        self._iterator: Any | None = None

    def open(self) -> None:
        from nvidia.dali import fn, pipeline_def, types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

        paths = [source.path for source in self._sources]
        labels = list(range(len(paths)))
        recipe = self._recipe

        @pipeline_def(
            batch_size=self._execution.batch_size,
            num_threads=max(1, self._execution.num_workers),
            device_id=0,
            seed=self._execution.seed,
            prefetch_queue_depth=self._execution.prefetch_factor or 2,
        )
        def pipeline_definition() -> Any:
            encoded, reader_labels = fn.readers.file(
                files=[str(path) for path in paths],
                labels=labels,
                # The caller supplies the shared per-seed permutation.  A
                # second reader-local shuffle would give DALI different
                # input order from every other implementation.
                random_shuffle=False,
                name="Reader",
            )
            data = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
            for stage in recipe.stages:
                data = _apply_stage(
                    data,
                    stage,
                    output_height=self._execution.height,
                    output_width=self._execution.width,
                )
            return data, reader_labels

        pipeline = pipeline_definition()
        pipeline.build()
        self._pipeline = pipeline
        self._iterator = iter(
            DALIGenericIterator(
                [pipeline],
                ["data", "label"],
                reader_name="Reader",
                auto_reset=False,
                last_batch_policy=LastBatchPolicy.DROP,
                prepare_first_batch=False,
            ),
        )

    def next_batch(self) -> tuple[Any, Any]:
        if self._iterator is None:
            raise RuntimeError("DALI source is not open")
        item = next(self._iterator)[0]
        data = item["data"]
        labels = item["label"].reshape(-1).long()
        return data, labels

    def close(self) -> None:
        self._iterator = None
        self._pipeline = None


def _apply_stage(data: Any, stage: RecipeStage, *, output_height: int, output_width: int) -> Any:
    handler = _STAGE_HANDLERS.get(stage.operation_id)
    if handler is None:
        raise UnsupportedRecipeError(f"DALI does not implement operation {stage.operation_id!r}")
    return handler(data, stage.parameters, output_height, output_width)


def _apply_to_tensor(data: Any, _params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    return data


def _apply_normalize(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn, types

    mean = [float(value) * 255.0 for value in _number_list(params, "mean")]
    std = [float(value) * 255.0 for value in _number_list(params, "std")]
    return fn.crop_mirror_normalize(
        data,
        dtype=types.FLOAT16,
        output_layout="CHW",
        mean=mean,
        std=std,
    )


def _apply_random_crop224(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn, types

    height = _integer(params, "height")
    width = _integer(params, "width")
    resized = fn.resize(data, resize_shorter=max(height, width), interp_type=types.INTERP_LINEAR)
    return fn.crop(
        resized,
        crop=(height, width),
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
    )


def _apply_random_resized_crop(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.random_resized_crop(
        data,
        size=tuple(_integer_list(params, "size")),
        random_area=tuple(_number_list(params, "scale")),
        random_aspect_ratio=tuple(_number_list(params, "ratio")),
        interp_type=_interpolation(params.get("interpolation")),
    )


def _apply_resize(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    target = _integer(params, "target_size")
    return fn.resize(
        data,
        resize_x=target,
        resize_y=target,
        interp_type=_interpolation(params.get("interpolation")),
    )


def _apply_horizontal_flip(data: Any, _params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.flip(data, horizontal=1)


def _apply_vertical_flip(data: Any, _params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.flip(data, vertical=1)


def _apply_pad(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn, types

    padding = _integer(params, "padding")
    return fn.slice(
        data,
        axes=(0, 1),
        start=(-padding, -padding),
        shape=data.shape(dtype=types.INT32)[:2] + 2 * padding,
        out_of_bounds_policy="pad",
        fill_values=params.get("fill", 0),
    )


def _apply_rotate(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.rotate(
        data,
        angle=_uniform(params.get("angle_range")),
        keep_size=True,
        fill_value=params.get("fill", 0),
        interp_type=_interpolation(params.get("interpolation")),
    )


def _apply_affine(data: Any, params: dict[str, Any], output_height: int, output_width: int) -> Any:
    from nvidia.dali import fn

    scale = _number(params, "scale")
    shift = _number_list(params, "shift")
    return fn.warp_affine(
        data,
        matrix=[1.0 / scale, 0.0, -shift[0], 0.0, 1.0 / scale, -shift[1]],
        size=(output_height, output_width),
        fill_value=params.get("fill", 0),
        interp_type=_interpolation(params.get("interpolation")),
    )


def _apply_shear(data: Any, params: dict[str, Any], output_height: int, output_width: int) -> Any:
    from nvidia.dali import fn

    shear = _number(params, "shear") / 57.29577951308232
    return fn.warp_affine(
        data,
        matrix=[1.0, shear, 0.0, 0.0, 1.0, 0.0],
        size=(output_height, output_width),
    )


def _apply_brightness(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    limit = _number_list(params, "brightness_limit")
    return fn.brightness_contrast(data, brightness=1.0 + limit[0])


def _apply_contrast(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    limit = _number_list(params, "contrast_limit")
    return fn.brightness_contrast(data, contrast=1.0 + limit[0])


def _apply_color_jitter(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.color_twist(
        data,
        brightness=_uniform(params.get("brightness", 1.0)),
        contrast=_uniform(params.get("contrast", 1.0)),
        saturation=_uniform(params.get("saturation", 1.0)),
        hue=_uniform(params.get("hue", 0.0)),
    )


def _apply_hue(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.hue(data, hue=_number(params, "hue"))


def _apply_saturation(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.saturation(data, saturation=1.0 + _number(params, "saturation_factor"))


def _apply_clahe(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    tile_grid = _integer_list(params, "tile_grid_size")
    return fn.clahe(
        data,
        clip_limit=_number_list(params, "clip_limit")[0],
        tiles_x=tile_grid[1],
        tiles_y=tile_grid[0],
    )


def _apply_equalize(data: Any, _params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.equalize(data)


def _apply_gaussian_blur(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.gaussian_blur(
        data,
        sigma=_number(params, "sigma"),
        window_size=tuple(_integer_list(params, "kernel_size")),
    )


def _apply_gaussian_noise(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.noise.gaussian(data, mean=_number(params, "mean"), stddev=_number(params, "std") * 255.0)


def _apply_salt_and_pepper(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.noise.salt_and_pepper(
        data,
        prob=_uniform(params.get("amount")),
        salt_vs_pepper=_uniform(params.get("salt_vs_pepper")),
    )


def _apply_erasing(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn, types

    anchor, shape = fn.random_crop_generator(
        data.shape(),
        random_area=tuple(_number_list(params, "scale")),
        random_aspect_ratio=tuple(_number_list(params, "ratio")),
    )
    return fn.erase(
        data,
        axis_names="HW",
        anchor=fn.cast(anchor, dtype=types.FLOAT),
        shape=fn.cast(shape, dtype=types.FLOAT),
        fill_value=params.get("fill", 0),
    )


def _apply_jpeg_compression(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.jpeg_compression_distortion(data, quality=_integer(params, "quality"))


_STAGE_HANDLERS = {
    "ToTensor": _apply_to_tensor,
    "Normalize": _apply_normalize,
    "RandomCrop224": _apply_random_crop224,
    "RandomResizedCrop": _apply_random_resized_crop,
    "Resize": _apply_resize,
    "HorizontalFlip": _apply_horizontal_flip,
    "VerticalFlip": _apply_vertical_flip,
    "Pad": _apply_pad,
    "Rotate": _apply_rotate,
    "Affine": _apply_affine,
    "Shear": _apply_shear,
    "Brightness": _apply_brightness,
    "Contrast": _apply_contrast,
    "ColorJitter": _apply_color_jitter,
    "ColorJiggle": _apply_color_jitter,
    "Hue": _apply_hue,
    "Saturation": _apply_saturation,
    "CLAHE": _apply_clahe,
    "Equalize": _apply_equalize,
    "GaussianBlur": _apply_gaussian_blur,
    "GaussianNoise": _apply_gaussian_noise,
    "SaltAndPepper": _apply_salt_and_pepper,
    "Erasing": _apply_erasing,
    "JpegCompression": _apply_jpeg_compression,
}


def _interpolation(value: Any) -> Any:
    from nvidia.dali import types

    return types.INTERP_NN if value == "nearest" else types.INTERP_LINEAR


def _uniform(value: Any) -> Any:
    from nvidia.dali import fn

    if isinstance(value, list):
        return fn.random.uniform(range=(float(value[0]), float(value[1])))
    return float(value)


def _number(params: dict[str, Any], name: str) -> float:
    value = params.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"DALI recipe parameter {name!r} must be numeric")
    return float(value)


def _integer(params: dict[str, Any], name: str) -> int:
    value = params.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"DALI recipe parameter {name!r} must be an integer")
    return value


def _number_list(params: dict[str, Any], name: str) -> list[float]:
    value = params.get(name)
    if not isinstance(value, list) or not value:
        raise TypeError(f"DALI recipe parameter {name!r} must be a non-empty numeric list")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise TypeError(f"DALI recipe parameter {name!r} must be a numeric list")
        result.append(float(item))
    return result


def _integer_list(params: dict[str, Any], name: str) -> list[int]:
    value = params.get(name)
    if not isinstance(value, list) or not value:
        raise TypeError(f"DALI recipe parameter {name!r} must be a non-empty integer list")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(f"DALI recipe parameter {name!r} must be an integer list")
        result.append(item)
    return result
