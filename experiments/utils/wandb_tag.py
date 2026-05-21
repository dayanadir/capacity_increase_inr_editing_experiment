from datetime import datetime
from typing import Dict


def build_tag(prefix: str, attributes: Dict) -> str:
    return (
        f"{prefix}-"
        + "".join([f"{name}:{value}-" for name, value in attributes.items()])
        + f"{datetime.now()}"
    )
