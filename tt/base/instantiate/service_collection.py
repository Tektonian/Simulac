from abc import ABC
from typing import Any, List, Tuple, MutableMapping, Type, TypeVar

from .descriptor import SyncDescriptor
from .instantiate import ServiceIdentifier


T = TypeVar("T")
S = TypeVar("S", bound=ServiceIdentifier)


class ServiceCollection:

    def __init__(
        self, entries: List[Tuple[Type[ServiceIdentifier], SyncDescriptor[Any]]]
    ):
        self._entries: MutableMapping[Type[ServiceIdentifier], Any] = {}

        for k, v in entries:
            self._entries[k] = v

    def set[T](
        self,
        identifier: Type[ServiceIdentifier],
        instance_or_descriptor: T | SyncDescriptor[T],
    ) -> T | SyncDescriptor[T]:
        self._entries[identifier] = instance_or_descriptor
        ret = self._entries.get(identifier)
        return ret

    def has(self, identifier: Type[ServiceIdentifier]) -> bool:
        ret = self._entries.get(identifier)
        return False if ret is None else True

    def get[I: ServiceIdentifier](
        self, identifier: Type[I]
    ) -> I | SyncDescriptor[I] | None:
        return self._entries.get(identifier)
