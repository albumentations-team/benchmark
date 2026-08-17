from augbench.adapters.runtime import PythonModuleAdapter
from augbench.adapters.sources import load_albumentationsx_rgb


def create_adapter(implementation_id: str) -> PythonModuleAdapter:
    return PythonModuleAdapter(
        implementation_id=implementation_id,
        recipe_module="augbench.implementations.albumentationsx_pipeline_impl",
        load_source=load_albumentationsx_rgb,
        placement="cpu",
        source_id="simplejpeg-rgb",
    )
