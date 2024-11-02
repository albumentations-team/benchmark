from typing import Any

import cv2
import numpy as np
from ultralytics.data.augment import (
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.utils.instance import Instances

# Ensure single thread
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class UltralyticsImpl:
    """Ultralytics implementations of transforms"""

    @staticmethod
    def _create_empty_instances(image: np.ndarray) -> Instances:
        """Create empty instances object for the image"""
        return Instances(
            bboxes=np.zeros((0, 4)),  # no boxes
            segments=np.zeros((0, 1000, 2)),  # no segments
            keypoints=None,  # no keypoints
            bbox_format="xyxy",  # format of the boxes
            normalized=True,  # whether coordinates are normalized
        )

    @staticmethod
    def HorizontalFlip(params: dict[str, Any]) -> Any:
        aug = RandomFlip(direction="horizontal", p=params["p"])
        return lambda x: aug({"img": x, "instances": UltralyticsImpl._create_empty_instances(x)})["img"]

    @staticmethod
    def VerticalFlip(params: dict[str, Any]) -> Any:
        aug = RandomFlip(direction="vertical", p=params["p"])
        return lambda x: aug({"img": x, "instances": UltralyticsImpl._create_empty_instances(x)})["img"]

    @staticmethod
    def RandomPerspective(params: dict[str, Any]) -> Any:
        """Create a RandomPerspective transform."""
        scale = params.get("scale", (0.05, 0.1))
        # Ensure scale is a tuple
        if not isinstance(scale, tuple):
            scale = (scale, scale)

        def transform(image: np.ndarray) -> np.ndarray:
            # Create empty instances with the correct shape
            instances = UltralyticsImpl._create_empty_instances(image)

            # Apply the transform
            aug = RandomPerspective(
                degrees=0.0,  # no rotation
                translate=0.0,  # no translation
                scale=scale[0],  # use first value of scale range
                shear=0.0,  # no shear
                perspective=0.0,  # no perspective
                border=(0, 0),  # no border
            )

            # Create labels dictionary with required fields
            labels = {
                "img": image,
                "cls": np.zeros((0,)),  # empty class array
                "instances": instances,
            }

            # Apply transform and return only the image
            result = aug(labels)
            return result["img"]

        return transform

    @staticmethod
    def HSV(params: dict[str, Any]) -> Any:
        aug = RandomHSV(
            hgain=params["hue"],
            sgain=params["saturation"],
            vgain=params["value"],
        )
        return lambda x: aug({"img": x})["img"]  # HSV doesn't need instances

    @staticmethod
    def __call__(transform: Any, image: np.ndarray) -> np.ndarray:
        """Apply the transform to the image"""
        result = transform(image)
        return np.ascontiguousarray(result)
