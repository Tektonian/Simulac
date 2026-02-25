from typing import Generic, Type, List, Any, TypeVar
from .instantiate import ServiceIdentifier

T = TypeVar("T", bound=ServiceIdentifier)


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
