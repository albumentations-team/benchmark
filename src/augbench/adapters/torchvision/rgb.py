from augbench.adapters.runtime import Placement, PythonModuleAdapter
from augbench.adapters.sources import load_torchvision_rgb


def create_adapter(implementation_id: str) -> PythonModuleAdapter:
    placement: Placement = "cuda" if implementation_id.endswith("_gpu") else "cpu"
    return PythonModuleAdapter(
        implementation_id=implementation_id,
        recipe_module="augbench.implementations.torchvision_pipeline_impl",
        load_source=load_torchvision_rgb,
        placement=placement,
        source_id="torchvision-io",
        unsupported_operation_ids=(frozenset({"JpegCompression"}) if placement == "cuda" else frozenset()),
    )
