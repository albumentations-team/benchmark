from augbench.adapters.runtime import Placement, PythonModuleAdapter
from augbench.adapters.sources import load_kornia_rgb

_CUDA_UNSUPPORTED = frozenset(
    {
        "CornerIllumination",
        "Erasing",
        "LinearIllumination",
        "Perspective",
        "Shear",
    },
)


def create_adapter(implementation_id: str) -> PythonModuleAdapter:
    placement: Placement = "cuda" if implementation_id.endswith("_gpu") else "cpu"
    return PythonModuleAdapter(
        implementation_id=implementation_id,
        recipe_module="augbench.implementations.kornia_pipeline_impl",
        load_source=load_kornia_rgb,
        placement=placement,
        source_id="torchvision-io-float32",
        unsupported_operation_ids=_CUDA_UNSUPPORTED if placement == "cuda" else frozenset(),
    )
