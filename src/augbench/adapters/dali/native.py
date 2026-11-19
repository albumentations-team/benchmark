from __future__ import annotations

from itertools import permutations
from math import pi
from typing import TYPE_CHECKING, Any, Protocol

from augbench.implementations.geometry import (
    PERSPECTIVE_JITTER_LIMIT,
    affine_axis_ranges,
    centered_inverse_affine,
    emboss_kernel,
    inverse_perspective_coefficients,
    perspective_corners,
    perspective_edges_squared,
)
from augbench.implementations.specs import odd_kernel_sizes, parameter_range
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
    jpeg_channels: int


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
        from nvidia.dali import fn, pipeline_def
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
            enable_conditionals=True,
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
            data, stages = _decode_recipe_input(encoded, recipe.stages, jpeg_channels=self._execution.jpeg_channels)
            for stage in stages:
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


def _decode_recipe_input(
    encoded: Any, stages: Sequence[RecipeStage], *, device: str = "mixed", jpeg_channels: int = 3
) -> tuple[Any, Sequence[RecipeStage]]:
    from nvidia.dali import fn, types

    decoder_args = {
        "device": device,
        "output_type": types.GRAY if jpeg_channels == 1 else types.RGB,
        "adjust_orientation": False,
    }
    first = stages[0]
    params = first.parameters
    if first.operation_id == "RandomResizedCrop":
        cropped = fn.decoders.image_random_crop(
            encoded,
            random_area=tuple(_number_list(params, "scale")),
            random_aspect_ratio=tuple(_number_list(params, "ratio")),
            **decoder_args,
        )
        return fn.resize(
            cropped,
            size=tuple(_integer_list(params, "size")),
            interp_type=_interpolation(params["interpolation"]),
            antialias=params["antialias"],
        ), stages[1:]
    if first.operation_id in {"RandomCrop224", "CenterCrop"}:
        return _decode_crop(encoded, params, random_crop=first.operation_id == "RandomCrop224", **decoder_args), stages[
            1:
        ]
    return fn.decoders.image(encoded, **decoder_args), stages


def _decode_crop(encoded: Any, params: dict[str, Any], *, random_crop: bool, **decoder_args: Any) -> Any:
    from nvidia.dali import fn, math, types

    target = types.Constant([_integer(params, "height"), _integer(params, "width")], dtype=types.INT32)
    shape = fn.peek_image_shape(encoded, adjust_orientation=False, dtype=types.INT32)[:2]
    extra = math.max(shape - target, 0)
    if random_crop:
        offsets = math.floor(fn.random.uniform(range=(0.0, 1.0), shape=[2]) * (extra + 1))
        start = fn.cast(offsets, dtype=types.INT32)
    else:
        start = extra // 2
    cropped = fn.decoders.image_slice(encoded, start, math.min(shape, target), axes=(0, 1), **decoder_args)
    return _apply_center_crop(cropped, params, 0, 0)


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
    offset = fn.cast(math.floor(fn.random.uniform(range=(0.0, 1.0), shape=[2]) * choices), dtype=types.INT32)
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


def _apply_channel_swap(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    order = _integer_list(params, "channel_order")
    if order == [2, 1, 0]:
        return fn.color_space_conversion(data, image_type=types.RGB, output_type=types.BGR)
    channels = [
        fn.slice(data, (index,), (1,), axes=(2,), normalized_anchor=False, normalized_shape=False) for index in order
    ]
    return fn.cat(*channels, axis=2)


def _apply_grayscale(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    gray = fn.color_space_conversion(data, image_type=types.RGB, output_type=types.GRAY)
    if params["num_output_channels"] == 1:
        return gray
    return fn.color_space_conversion(gray, image_type=types.GRAY, output_type=types.RGB)


def _apply_channel_shuffle(data: Any, _params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    orders = tuple(permutations(range(3)))
    choices = list(range(len(orders)))
    order = fn.random.uniform(values=choices, shape=[1], dtype=types.INT32)
    channels = []
    for channel_index in range(3):
        values = [permutation[channel_index] for permutation in orders]
        start = fn.lookup_table(order, keys=choices, values=values, dtype=types.INT32)
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
    from nvidia.dali import fn, types

    low, high = parameter_range(params["bits"])
    masks = [(255 << (8 - bits)) & 255 for bits in range(low, high + 1)]
    mask = fn.random.uniform(values=masks, dtype=types.UINT8, device=data.device)
    return data & mask


def _apply_solarize(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    threshold = 255 * _uniform(params["threshold"], device=data.device)
    keep = fn.cast(data < threshold, dtype=types.UINT8)
    return fn.cast(data * keep + (255 - data) * (1 - keep), dtype=types.UINT8)


def _apply_random_gamma(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, math, types

    gamma = _uniform(params["gamma"], device=data.device) / 100
    return fn.cast(math.floor(255 * math.pow(data / 255.0, gamma)), dtype=types.UINT8)


def _apply_median_blur(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    size = fn.random.uniform(values=list(odd_kernel_sizes(params["blur_limit"])), dtype=types.INT32)
    return fn.experimental.median_blur(data, window_size=fn.stack(size, size))


def _apply_blur(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn, types

    sizes = list(odd_kernel_sizes(params["kernel_size"]))
    size = fn.random.uniform(values=sizes, dtype=types.INT32)
    ones = types.Constant(1.0, shape=(sizes[-1], sizes[-1]))
    kernel = fn.slice(
        ones,
        types.Constant([0, 0], dtype=types.INT32),
        fn.stack(size, size),
        axes=(0, 1),
        normalized_anchor=False,
        normalized_shape=False,
    ) / (size * size)
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
    from nvidia.dali import fn

    values = emboss_kernel(_uniform(params["alpha"]), _uniform(params["strength"]))
    kernel = fn.reshape(fn.stack(*values), shape=[3, 3])
    return fn.filter(data, kernel, border="reflect_101")


def _apply_pixel_spread(data: Any, params: dict[str, Any], height: int, width: int) -> Any:
    from nvidia.dali import fn, types

    radius = _integer(params, "radius")
    offsets = list(range(-radius, radius + 1))
    dx = fn.random.uniform(values=offsets, shape=(height, width), dtype=types.FLOAT, device=data.device)
    dy = fn.random.uniform(values=offsets, shape=(height, width), dtype=types.FLOAT, device=data.device)
    x = types.Constant(list(range(width)), shape=(1, width), dtype=types.FLOAT, device=data.device)
    y = types.Constant(list(range(height)), shape=(height, 1), dtype=types.FLOAT, device=data.device)
    return fn.experimental.remap(data, x + dx, y + dy, interp=types.INTERP_NN, pixel_origin="center")


def _apply_colorize(data: Any, params: dict[str, Any], _height: int, _width: int) -> Any:
    from nvidia.dali import fn

    black_values, white_values = _integer_list(params, "black"), _integer_list(params, "white")
    channels = [
        _byte_lookup(
            data,
            [
                int(black_values[index] + (white_values[index] - black_values[index]) * value / 255)
                for value in range(256)
            ],
        )
        for index in range(3)
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
    from nvidia.dali import fn, math, types

    x_range, y_range = affine_axis_ranges(params["shift"])
    x = fn.random.uniform(values=list(range(x_range[0], x_range[1] + 1)), dtype=types.INT32)
    y = fn.random.uniform(values=list(range(y_range[0], y_range[1] + 1)), dtype=types.INT32)
    sx = sy = 0.0
    if "shear" in params:
        sx_range, sy_range = affine_axis_ranges(params["shear"])
        sx = math.tan(_uniform(sx_range) * (pi / 180))
        sy = math.tan(_uniform(sy_range) * (pi / 180))
    angle = _uniform(params["angle"]) * (pi / 180)
    matrix = centered_inverse_affine(
        (output_width / 2, output_height / 2),
        (math.sin(angle), math.cos(angle)),
        _uniform(params["scale"]),
        (x, y),
        (sx, sy),
    )
    return fn.warp_affine(
        data,
        fn.reshape(fn.stack(*matrix), shape=(2, 3)),
        size=(output_height, output_width),
        inverse_map=True,
        fill_value=params.get("fill", 0),
        interp_type=_interpolation(params.get("interpolation", "bilinear")),
    )


def _apply_shear(data: Any, params: dict[str, Any], output_height: int, output_width: int) -> Any:
    from nvidia.dali import fn, math, types

    slope = -math.tan(_uniform(params["shear"]) * (pi / 180))
    zero = types.Constant(0.0)
    one = types.Constant(1.0)
    coefficients = fn.stack(one, slope, -slope * (output_height / 2), zero, one, zero)
    return fn.warp_affine(
        data,
        fn.reshape(coefficients, shape=(2, 3)),
        size=(output_height, output_width),
        inverse_map=True,
        fill_value=0,
        interp_type=types.INTERP_LINEAR,
    )


def _apply_perspective(data: Any, params: dict[str, Any], height: int, width: int) -> Any:
    from nvidia.dali import fn, math, types

    jitter = math.abs(fn.random.normal(shape=(4, 2)) * _uniform(params["scale"]))
    jitter = jitter - math.floor(jitter / PERSPECTIVE_JITTER_LIMIT) * PERSPECTIVE_JITTER_LIMIT
    corners = perspective_corners(tuple((jitter[i, 0], jitter[i, 1]) for i in range(4)), height, width)
    widths, heights = perspective_edges_squared(corners)
    sx = math.floor(math.sqrt(math.max(*widths))) / width**2
    sy = math.floor(math.sqrt(math.max(*heights))) / height**2
    coefficients = inverse_perspective_coefficients(corners, sx, sy)
    matrix = fn.reshape(fn.stack(*coefficients, types.Constant(1.0), axis=0), shape=[3, 3])
    return fn.experimental.warp_perspective(
        data, matrix.gpu(), size=(height, width), pixel_origin="center", fill_value=[params["fill"]]
    )


def _apply_brightness(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.brightness_contrast(data, brightness=1.0 + _uniform(params["brightness_limit"]))


def _apply_contrast(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    return _contrast(data, 1 + _uniform(params["contrast_limit"], device=data.device))


def _apply_color_jitter(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn, types

    brightness = _uniform(params["brightness"])
    contrast = _uniform(params["contrast"])
    saturation = _uniform(params["saturation"])
    hue = 360 * _uniform(params["hue"])
    orders = tuple(permutations(range(4)))
    keys = list(range(len(orders)))
    order = fn.random.uniform(values=keys, dtype=types.INT32)
    brightness_step, contrast_step, saturation_step = range(3)
    for permutation_index, operations in enumerate(orders):
        if order == permutation_index:
            for operation in operations:
                if operation == brightness_step:
                    data = fn.brightness(data, brightness=brightness)
                elif operation == contrast_step:
                    data = _contrast(data, contrast.gpu())
                elif operation == saturation_step:
                    data = _saturation(data, saturation.gpu())
                else:
                    data = fn.hue(data, hue=hue)
    return data


def _apply_hue(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.hue(data, hue=_uniform(params["hue_degrees"]))


def _apply_saturation(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    return _saturation(data, 1 + _uniform(params["saturation_factor"], device=data.device))


def _contrast(data: Any, factor: Any) -> Any:
    from nvidia.dali import fn, types

    gray = fn.color_space_conversion(data, image_type=types.RGB, output_type=types.GRAY)
    center = fn.reductions.mean(gray)
    return fn.cast(center + factor * (data - center), dtype=types.UINT8)


def _saturation(data: Any, factor: Any) -> Any:
    from nvidia.dali import fn, types

    gray = fn.color_space_conversion(data, image_type=types.RGB, output_type=types.GRAY)
    return fn.cast(factor * data + (1 - factor) * gray, dtype=types.UINT8)


def _apply_clahe(data: Any, params: dict[str, Any], height: int, width: int) -> Any:
    from nvidia.dali import fn, math

    tiles_y, tiles_x = _integer_list(params, "tile_grid_size")
    low, high = _number_list(params, "clip_range")
    tile_height = (height + tiles_y - 1) // tiles_y
    tile_width = (width + tiles_x - 1) // tiles_x
    pixels_per_bin = tile_height * tile_width / 256
    # DALI needs a static argument, but its histogram clamps to an integer count.
    count = math.floor(_uniform(params["clip_range"]) * pixels_per_bin)
    for threshold in range(int(low * pixels_per_bin), int(high * pixels_per_bin) + 1):
        if count == threshold:
            data = fn.clahe(data, clip_limit=(threshold + 0.5) / pixels_per_bin, tiles_x=tiles_x, tiles_y=tiles_y)
    return data


def _apply_equalize(data: Any, _params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.equalize(data)


def _apply_gaussian_blur(data: Any, params: dict[str, Any], _output_height: int, _output_width: int) -> Any:
    from nvidia.dali import fn

    return fn.gaussian_blur(
        data,
        sigma=_uniform(params["sigma"]),
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
    from nvidia.dali import fn, types

    low, high = _integer_list(params, "quality_range")
    quality = fn.random.uniform(values=list(range(low, high + 1)), dtype=types.INT32)
    return fn.jpeg_compression_distortion(data, quality=quality)


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
    "ShiftScaleRotate": _apply_affine,
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
    "Perspective": _apply_perspective,
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


def _uniform(value: Any, *, device: str = "cpu") -> Any:
    from nvidia.dali import fn, types

    low, high = parameter_range(value)
    if low == high:
        return types.Constant(float(low), device=device)
    return fn.random.uniform(range=(float(low), float(high)), device=device)


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
