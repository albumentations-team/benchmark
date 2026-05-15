from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as tv_transforms

LIBRARY = "pytorchvideo"
RANDOMNESS_SCOPE = "per_clip"


class _ToFloat:
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        video = video.contiguous()
        if video.dtype == torch.uint8:
            return video.float().div(255.0)
        return video.float() if not video.is_floating_point() else video


class _UniformTemporalSubsample:
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        frame_count = video.shape[1]
        indices = torch.linspace(0, frame_count - 1, self.num_samples, device=video.device).long()
        return torch.index_select(video, 1, indices).contiguous()


class _Normalize:
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=video.device, dtype=video.dtype)
        std = self.std.to(device=video.device, dtype=video.dtype)
        return (video - mean) / std


class _ShortSideScale:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, height, width = video.shape
        if height < width:
            new_height = self.size
            new_width = round(width * self.size / height)
        else:
            new_width = self.size
            new_height = round(height * self.size / width)
        frames = video.permute(1, 0, 2, 3)
        scaled = F.interpolate(frames, size=(new_height, new_width), mode="bilinear", align_corners=False)
        return scaled.permute(1, 0, 2, 3).contiguous()


def __call__(transform: Any, video: Any) -> Any:  # noqa: N807
    return transform(video).contiguous()


def _canonical_recipe() -> Any:
    return tv_transforms.Compose(
        [
            _UniformTemporalSubsample(16),
            _ToFloat(),
            _Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            _ShortSideScale(256),
            tv_transforms.RandomCrop((224, 224)),
            tv_transforms.RandomHorizontalFlip(p=0.5),
        ],
    )


TRANSFORMS: list[dict[str, Any]] = [
    {
        "name": "PyTorchVideoCanonical+Normalize+ToTensor",
        "transform": _canonical_recipe(),
    },
]
