"""Kornia multi-channel image benchmark spec.

Tests transforms on 9-channel images (3x stacked RGB). Run with --num-channels 9.
Excludes RGB-only transforms (ColorJitter, Hue, Saturation, CLAHE, Equalize, etc.).
"""

from typing import Any

import kornia.augmentation as Kaug
import torch

LIBRARY = "kornia"
NUM_CHANNELS = 9

torch.set_num_threads(1)


def __call__(transform: Any, image: Any) -> Any:  # noqa: N807
    return transform(image.unsqueeze(0)).squeeze(0)


TRANSFORMS = [
    {"name": "HorizontalFlip", "transform": Kaug.RandomHorizontalFlip(p=1)},
    {"name": "VerticalFlip", "transform": Kaug.RandomVerticalFlip(p=1)},
    {"name": "RandomRotate90", "transform": Kaug.RandomRotation90(times=(0, 3), p=1)},
    {"name": "Rotate", "transform": Kaug.RandomRotation(degrees=(45, 45), p=1)},
    {
        "name": "Affine",
        "transform": Kaug.RandomAffine(
            degrees=(25.0, 25.0),
            translate=(0.04, 0.04),
            scale=(2.0, 2.0),
            shear=(10.0, 15.0),
            p=1,
        ),
    },
    {"name": "Perspective", "transform": Kaug.RandomPerspective(distortion_scale=0.1, p=1)},
    {"name": "Shear", "transform": Kaug.RandomShear(shear=10, p=1)},
    {
        "name": "ThinPlateSpline",
        "transform": Kaug.RandomThinPlateSpline(scale=0.5, p=1),
    },
    {
        "name": "OpticalDistortion",
        "transform": Kaug.RandomFisheye(
            center_x=torch.tensor([-0.3, 0.3]),
            center_y=torch.tensor([-0.3, 0.3]),
            gamma=torch.tensor([0.9, 1.1]),
            p=1,
        ),
    },
    {
        "name": "Elastic",
        "transform": Kaug.RandomElasticTransform(
            sigma=(5.0, 5.0),
            alpha=(50.0, 50.0),
            p=1,
        ),
    },
    {"name": "RandomCrop128", "transform": Kaug.RandomCrop(size=(128, 128), pad_if_needed=True, p=1)},
    {"name": "CenterCrop128", "transform": Kaug.CenterCrop(size=(128, 128), p=1)},
    {
        "name": "RandomResizedCrop",
        "transform": Kaug.RandomResizedCrop(
            size=(512, 512),
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            p=1,
        ),
    },
    {"name": "Resize", "transform": Kaug.Resize(size=(512, 512), p=1)},
    {"name": "LongestMaxSize", "transform": Kaug.LongestMaxSize(max_size=512, p=1)},
    {"name": "SmallestMaxSize", "transform": Kaug.SmallestMaxSize(max_size=512, p=1)},
    {"name": "RandomJigsaw", "transform": Kaug.RandomJigsaw(grid=(4, 4), p=1)},
    {"name": "Invert", "transform": Kaug.RandomInvert(p=1)},
    {"name": "Posterize", "transform": Kaug.RandomPosterize(bits=4.0, p=1)},
    {"name": "Solarize", "transform": Kaug.RandomSolarize(thresholds=0.5, p=1)},
    {"name": "RandomGamma", "transform": Kaug.RandomGamma(gamma=(1.2, 1.2), p=1)},
    {"name": "GaussianNoise", "transform": Kaug.RandomGaussianNoise(mean=0, std=0.44, p=1)},
    {"name": "Erasing", "transform": Kaug.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0, p=1)},
    {"name": "Blur", "transform": Kaug.RandomBoxBlur(kernel_size=(5, 5), border_type="constant", p=1)},
    {"name": "GaussianBlur", "transform": Kaug.RandomGaussianBlur(kernel_size=(5, 5), sigma=(2.0, 2.0), p=1)},
    {"name": "MotionBlur", "transform": Kaug.RandomMotionBlur(kernel_size=5, angle=(0, 360), direction=(-1, 1), p=1)},
    {"name": "Sharpen", "transform": Kaug.RandomSharpness(sharpness=2.0, p=1)},
    {"name": "AutoContrast", "transform": Kaug.RandomAutoContrast(p=1)},
    {"name": "Brightness", "transform": Kaug.RandomBrightness(brightness=(1.2, 1.2), p=1)},
    {"name": "Contrast", "transform": Kaug.RandomContrast(contrast=(1.2, 1.2), p=1)},
    {
        "name": "Normalize",
        "transform": Kaug.Normalize(
            mean=(0.485, 0.456, 0.406) * 3,
            std=(0.229, 0.224, 0.225) * 3,
            p=1,
        ),
    },
    {"name": "ChannelShuffle", "transform": Kaug.RandomChannelShuffle(p=1)},
    {"name": "ChannelDropout", "transform": Kaug.RandomChannelDropout(p=1)},
    {"name": "LinearIllumination", "transform": Kaug.RandomLinearIllumination(gain=(0.01, 0.2), p=1)},
    {"name": "CornerIllumination", "transform": Kaug.RandomLinearCornerIllumination(gain=(0.01, 0.2), p=1)},
    {"name": "GaussianIllumination", "transform": Kaug.RandomGaussianIllumination(gain=(0.01, 0.2), p=1)},
    {"name": "PlasmaBrightness", "transform": Kaug.RandomPlasmaBrightness(roughness=(0.5, 0.5), p=1)},
    {"name": "PlasmaContrast", "transform": Kaug.RandomPlasmaContrast(roughness=(0.5, 0.5), p=1)},
    {"name": "PlasmaShadow", "transform": Kaug.RandomPlasmaShadow(roughness=(0.5, 0.5), p=1)},
]
