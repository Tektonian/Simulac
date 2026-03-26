from typing import Any, Generic, List, Type, TypeVar

from .instantiate import ServiceIdentifier

T = TypeVar("T", bound=ServiceIdentifier[object])


class SyncDescriptor(Generic[T]):
    def __init__(
        self,
        ctor: Type[T] | object,
        static_arguments: List[Any],
        support_delayed_instantiation: bool,
    ):
        self.ctor = ctor
        self.static_arguments = static_arguments
        self.support_delayed_instantiation = support_delayed_instantiation
