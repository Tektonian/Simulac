from typing import Generic, Tuple, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")

# Type alias for convenience
ResultType = Union[Tuple[T, None], Tuple[None, E]]
