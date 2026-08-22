from augbench.adapters.runtime import PythonModuleAdapter, PythonModuleAdapterConfig
from augbench.adapters.sources import load_pillow_rgb


def create_adapter(implementation_id: str) -> PythonModuleAdapter:
    return PythonModuleAdapter(
        config=PythonModuleAdapterConfig(
            implementation_id=implementation_id,
            recipe_module="augbench.implementations.pillow_pipeline_impl",
            placement="cpu",
            source_id="pillow",
        ),
        load_source=load_pillow_rgb,
    )
