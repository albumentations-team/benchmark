from augbench.adapters.runtime import Placement, PythonModuleAdapter, PythonModuleAdapterConfig
from augbench.adapters.sources import load_kornia_rgb


def create_adapter(implementation_id: str) -> PythonModuleAdapter:
    placement: Placement = "cuda" if implementation_id.endswith("_gpu") else "cpu"
    return PythonModuleAdapter(
        config=PythonModuleAdapterConfig(
            implementation_id=implementation_id,
            recipe_module="augbench.implementations.kornia_pipeline_impl",
            placement=placement,
            source_id="torchvision-io-float32",
        ),
        load_source=load_kornia_rgb,
    )
