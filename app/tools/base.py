from typing import Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str

