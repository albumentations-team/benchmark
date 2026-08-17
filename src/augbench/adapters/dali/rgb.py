from augbench.adapters.dali.common import DaliAdapter


def create_adapter(implementation_id: str) -> DaliAdapter:
    return DaliAdapter(implementation_id)
