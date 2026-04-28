"""Torchvision multi-channel image benchmark spec.

Tests transforms on 9-channel images (3x stacked RGB). Run with --num-channels 9.
Excludes RGB-only transforms (ColorJitter, Hue, Saturation, CLAHE, Equalize, etc.).
"""

from typing import Any

import torch
import torchvision.transforms.v2 as tv_transforms

LIBRARY = "torchvision"
NUM_CHANNELS = 9

torch.set_num_threads(1)


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image)


TRANSFORMS = [
    {"name": "HorizontalFlip", "transform": tv_transforms.RandomHorizontalFlip(p=1)},
    {"name": "VerticalFlip", "transform": tv_transforms.RandomVerticalFlip(p=1)},
    {
        "name": "RandomRotate90",
        "transform": lambda image: torch.rot90(image, int(torch.randint(0, 4, ()).item()), dims=(-2, -1)),
    },
    {
        "name": "Rotate",
        "transform": tv_transforms.RandomRotation(
            degrees=45,
            interpolation=tv_transforms.InterpolationMode.NEAREST,
            fill=0,
        ),
    },
    {
        "name": "Affine",
        "transform": tv_transforms.RandomAffine(
            degrees=25.0,
            translate=(20 / 512, 20 / 512),
            scale=(2.0, 2.0),
            shear=(10.0, 15.0),
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
        ),
    },
    {
        "name": "Perspective",
        "transform": tv_transforms.RandomPerspective(
            distortion_scale=0.1,
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            fill=0,
            p=1,
        ),
    },
    {
        "name": "Shear",
        "transform": tv_transforms.RandomAffine(
            degrees=0,
            shear=10,
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
        ),
    },
    {
        "name": "Elastic",
        "transform": tv_transforms.ElasticTransform(
            alpha=50.0,
            sigma=5.0,
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
        ),
    },
    {"name": "RandomCrop128", "transform": tv_transforms.RandomCrop(size=(128, 128), pad_if_needed=True)},
    {"name": "CenterCrop128", "transform": tv_transforms.CenterCrop(size=(128, 128))},
    {
        "name": "RandomResizedCrop",
        "transform": tv_transforms.RandomResizedCrop(
            size=(512, 512),
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
        ),
    },
    {
        "name": "Resize",
        "transform": tv_transforms.Resize(
            size=512,
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
    },
    {"name": "Pad", "transform": tv_transforms.Pad(padding=10, fill=0, padding_mode="constant")},
    {"name": "Invert", "transform": tv_transforms.RandomInvert(p=1)},
    {"name": "Posterize", "transform": tv_transforms.RandomPosterize(bits=4, p=1)},
    {"name": "Solarize", "transform": tv_transforms.RandomSolarize(threshold=0.5, p=1)},
    {"name": "GaussianBlur", "transform": tv_transforms.GaussianBlur(kernel_size=(5, 5), sigma=(2.0, 2.0))},
    {
        "name": "Normalize",
        "transform": tv_transforms.Compose(
            [
                tv_transforms.ConvertImageDtype(torch.float32),
                tv_transforms.Normalize(mean=(0.485, 0.456, 0.406) * 3, std=(0.229, 0.224, 0.225) * 3),
            ],
        ),
    },
    {"name": "Erasing", "transform": tv_transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, p=1)},
    {"name": "ChannelShuffle", "transform": tv_transforms.RandomChannelPermutation()},
]
