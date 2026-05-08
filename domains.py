from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    data: dict
    def __init__(self) -> None:
        pass
    def __hash__(self) -> int:
        return hash(self)
    def __eq__(self, value: object, /) -> bool:
        return self == value

