from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING, Any, Protocol

from augbench.implementations.geometry import centered_inverse_affine, emboss_kernel
from augbench.implementations.shear import x_shear_matrix
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
    from nvidia.dali import fn, math, types

    height = _integer(params, "height")
    width = _integer(params, "width")
    extent = types.Constant([height, width], dtype=types.INT32)
    delta = data.shape(dtype=types.INT32)[:2] - extent
    choices = math.max(delta, 0) + 1
    offset = fn.cast(fn.random.uniform(range=(0.0, 1.0), shape=[2]) * choices, dtype=types.INT32)
    start = offset - math.max(-delta, 0) // 2
    return fn.slice(
        data, start, extent, axes=(0, 1), normalized_anchor=False, normalized_shape=False, out_of_bounds_policy="pad"
    )


def _apply_center_crop(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, math, types

    extent = types.Constant([_integer(params, "height"), _integer(params, "width")], dtype=types.INT32)
    delta = data.shape(dtype=types.INT32)[:2] - extent
    start = math.max(delta, 0) // 2 - math.max(-delta, 0) // 2
    return fn.slice(
        data, start, extent, axes=(0, 1), normalized_anchor=False, normalized_shape=False, out_of_bounds_policy="pad"
    )


def _apply_crop(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn

    return fn.slice(
        data,
        start=(_integer(params, "top"), _integer(params, "left")),
        shape=(_integer(params, "height"), _integer(params, "width")),
        axes=(0, 1),
        out_of_bounds_policy="pad",
    )


def _apply_crop_and_pad(data: Any, params: dict[str, Any], height: int, width: int) -> Any:
    from nvidia.dali import fn

    top, _, _, left = _integer_list(params, "sides")
    return fn.slice(data, start=(-top, -left), shape=(height, width), axes=(0, 1), out_of_bounds_policy="pad")


def _apply_letter_box(data: Any, params: dict[str, Any], height: int, width: int) -> Any:
    from nvidia.dali import fn, types

    fitted = fn.resize(data, size=(height, width), mode="not_larger", interp_type=types.INTERP_LINEAR, antialias=False)
    return _apply_center_crop(fitted, params, height, width)


def _apply_pad_if_needed(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, math, types

    minimum = types.Constant([_integer(params, "min_height"), _integer(params, "min_width")], dtype=types.INT32)
    extent = math.max(data.shape(dtype=types.INT32)[:2], minimum)
    return fn.slice(data, (0, 0), extent, axes=(0, 1), normalized_shape=False, out_of_bounds_policy="pad")


def _apply_longest_max_size(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn

    return fn.resize(
        data,
        resize_longer=_integer(params, "max_size"),
        interp_type=_interpolation(params["interpolation"]),
        antialias=False,
    )


def _apply_smallest_max_size(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn

    return fn.resize(
        data,
        resize_shorter=_integer(params, "max_size"),
        interp_type=_interpolation(params["interpolation"]),
        antialias=False,
    )


def _apply_random_scale(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, math

    shape = data.shape()[:2]
    scale = _uniform(params["scale"])
    return fn.resize(
        data, size=math.floor(shape * scale), interp_type=_interpolation(params["interpolation"]), antialias=False
    )


def _apply_channel_swap(data: Any, _params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    return fn.color_space_conversion(data, image_type=types.RGB, output_type=types.BGR)


def _apply_grayscale(data: Any, _params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    gray = fn.color_space_conversion(data, image_type=types.RGB, output_type=types.GRAY)
    return fn.color_space_conversion(gray, image_type=types.GRAY, output_type=types.RGB)


def _apply_channel_shuffle(data: Any, _params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    choices = list(range(6))
    order = fn.random.uniform(values=choices, shape=[1], dtype=types.INT32)
    channels = []
    for values in zip(*permutations(range(3)), strict=True):
        start = fn.lookup_table(order, keys=choices, values=list(values), dtype=types.INT32)
        channels.append(fn.slice(data, start, (1,), axes=(2,), normalized_anchor=False, normalized_shape=False))
    return fn.cat(*channels, axis=2)


def _apply_channel_dropout(data: Any, _params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn

    channel = fn.random.uniform(values=[0.0, 1.0, 2.0], shape=[1])
    return fn.erase(data, axis_names="C", anchor=channel, shape=[1.0], fill_value=0)


def _apply_rgb_shift(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    limit = _number(params, "pixel_shift")
    channels = [
        fn.slice(data, (index,), (1,), axes=(2,), normalized_anchor=False, normalized_shape=False)
        + fn.random.uniform(range=(-limit, limit)).gpu()
        for index in range(3)
    ]
    return fn.cast(fn.cat(*channels, axis=2), dtype=types.UINT8)


def _byte_lookup(data: Any, values: list[int]) -> Any:
    from nvidia.dali import fn, types

    return fn.lookup_table(data, keys=list(range(256)), values=values, dtype=types.UINT8)


def _apply_invert(data: Any, _params: dict[str, Any], _height: int, _width: int) -> Any:
    return _byte_lookup(data, list(range(255, -1, -1)))


def _apply_posterize(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    mask = (255 << (8 - _integer(params, "bits"))) & 255
    return _byte_lookup(data, [value & mask for value in range(256)])


def _apply_solarize(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    threshold = 255 * _number(params, "threshold")
    return _byte_lookup(data, [value if value < threshold else 255 - value for value in range(256)])


def _apply_random_gamma(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    gamma = _number(params, "gamma") / 100
    return _byte_lookup(data, [int(255 * (value / 255) ** gamma) for value in range(256)])


def _apply_median_blur(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn

    size = _integer(params, "blur_limit")
    return fn.experimental.median_blur(data, window_size=(size, size))


def _apply_blur(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    size = _integer(params, "kernel_size")
    kernel = types.Constant([1 / (size * size)] * (size * size), shape=[size, size], dtype=types.FLOAT)
    return fn.filter(data, kernel, border="reflect_101")


def _apply_transpose(data: Any, _params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn

    return fn.transpose(data, perm=(1, 0, 2), transpose_layout=False)


def _apply_random_rotate90(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    start, end = _integer_list(params, "times")
    angle = fn.random.uniform(values=[90 * turns for turns in range(start, end + 1)])
    return fn.rotate(data, angle=angle, keep_size=True, interp_type=types.INTERP_NN)


def _apply_random_sized_crop(data: Any, params: dict[str, Any], height: int, width: int) -> Any:
    from nvidia.dali import fn, types

    crop = _integer(params, "crop_size")
    cropped = _apply_random_crop224(data, {"height": crop, "width": crop}, height, width)
    return fn.resize(cropped, size=(height, width), interp_type=types.INTERP_LINEAR, antialias=False)


def _apply_coarse_dropout(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, math

    size = _integer(params, "hole_size")
    choices = _integer(params, "input_size") - size + 1
    anchor = math.floor(fn.random.uniform(range=(0.0, 1.0), shape=[2]) * choices)
    return fn.erase(data, axis_names="HW", anchor=anchor, shape=(size, size), fill_value=0)


def _apply_emboss(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    kernel = emboss_kernel(_number(params, "alpha"), _number(params, "strength"))
    return fn.filter(data, types.Constant(list(kernel), shape=[3, 3], dtype=types.FLOAT), border="reflect_101")


def _apply_pixel_spread(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn

    return fn.jitter(data, nDegree=2 * _integer(params, "radius"), fill_value=0)


def _apply_shift_scale_rotate(data: Any, params: dict[str, Any], height: int, width: int) -> Any:
    from nvidia.dali import fn, types

    x, y = _number_list(params, "shift")
    matrix = centered_inverse_affine(
        (width / 2, height / 2), _number(params, "angle"), _number(params, "scale"), (x, y)
    )
    return fn.warp_affine(
        data, matrix=list(matrix), size=(height, width), inverse_map=True, fill_value=0, interp_type=types.INTERP_LINEAR
    )


def _apply_colorize(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    gray = fn.color_space_conversion(data, image_type=types.RGB, output_type=types.GRAY)
    channels = [
        _byte_lookup(gray, [int(black + (white - black) * value / 255) for value in range(256)])
        for black, white in zip(_integer_list(params, "black"), _integer_list(params, "white"), strict=True)
    ]
    return fn.cat(*channels, axis=2)


def _apply_morphological(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, math

    size = _integer(params, "size")
    if params["operation"] == "dilation":
        dilate = fn.experimental.dilate
        return math.max(dilate(data, mask_size=(1, size)), dilate(data, mask_size=(size, 1)))
    erode = fn.experimental.erode
    return math.min(erode(data, mask_size=(1, size)), erode(data, mask_size=(size, 1)))


def _apply_random_resized_crop(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.random_resized_crop(
        data,
        size=tuple(_integer_list(params, "size")),
        random_area=tuple(_number_list(params, "scale")),
        random_aspect_ratio=tuple(_number_list(params, "ratio")),
        interp_type=_interpolation(params.get("interpolation")),
        antialias=params.get("antialias", True),
    )


def _apply_resize(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    target = _integer(params, "target_size")
    return fn.resize(
        data,
        resize_x=target,
        resize_y=target,
        interp_type=_interpolation(params.get("interpolation")),
        antialias=params.get("antialias", True),
    )


def _apply_horizontal_flip(data: Any, _params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.flip(data, horizontal=1)


def _apply_vertical_flip(data: Any, _params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.flip(data, horizontal=0, vertical=1)


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
    from nvidia.dali import fn, types

    return fn.warp_affine(
        data,
        matrix=list(x_shear_matrix(-_number(params, "shear"), output_height / 2)),
        size=(output_height, output_width),
        inverse_map=True,
        fill_value=0,
        interp_type=types.INTERP_LINEAR,
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
    "CenterCrop": _apply_center_crop,
    "Crop": _apply_crop,
    "CropAndPad": _apply_crop_and_pad,
    "LetterBox": _apply_letter_box,
    "PadIfNeeded": _apply_pad_if_needed,
    "LongestMaxSize": _apply_longest_max_size,
    "SmallestMaxSize": _apply_smallest_max_size,
    "RandomScale": _apply_random_scale,
    "RandomSizedCrop": _apply_random_sized_crop,
    "CoarseDropout": _apply_coarse_dropout,
    "Emboss": _apply_emboss,
    "PixelSpread": _apply_pixel_spread,
    "ShiftScaleRotate": _apply_shift_scale_rotate,
    "Colorize": _apply_colorize,
    "Morphological": _apply_morphological,
    "RandomResizedCrop": _apply_random_resized_crop,
    "Resize": _apply_resize,
    "HorizontalFlip": _apply_horizontal_flip,
    "VerticalFlip": _apply_vertical_flip,
    "Transpose": _apply_transpose,
    "RandomRotate90": _apply_random_rotate90,
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
    "MedianBlur": _apply_median_blur,
    "Blur": _apply_blur,
    "ChannelSwap": _apply_channel_swap,
    "ChannelShuffle": _apply_channel_shuffle,
    "ChannelDropout": _apply_channel_dropout,
    "RGBShift": _apply_rgb_shift,
    "Grayscale": _apply_grayscale,
    "Invert": _apply_invert,
    "Posterize": _apply_posterize,
    "Solarize": _apply_solarize,
    "RandomGamma": _apply_random_gamma,
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
