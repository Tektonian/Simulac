from typing import Any, List, MutableMapping, Tuple, Type, TypeVar

from .descriptor import SyncDescriptor
from .instantiate import ServiceIdentifier

T = TypeVar("T")


class ServiceCollection:
    def __init__(
        self,
        entries: List[
            Tuple[
                Type[ServiceIdentifier[object]],
                SyncDescriptor[ServiceIdentifier[object]],
            ]
        ],
    ):
        self._entries: MutableMapping[
            Type[ServiceIdentifier[object]],
            SyncDescriptor[ServiceIdentifier[object]] | object,
        ] = {}

        for k, v in entries:
            self._entries[k] = v

    def set(
        self,
        identifier: Type[ServiceIdentifier[object]],
        instance_or_descriptor: object | SyncDescriptor[ServiceIdentifier[object]],
    ) -> object | SyncDescriptor[ServiceIdentifier[object]] | None:
        self._entries[identifier] = instance_or_descriptor
        ret: SyncDescriptor[ServiceIdentifier[object]] | object = self._entries.get(
            identifier
        )
        return ret

    def has(self, identifier: Type[ServiceIdentifier[Any]]) -> bool:
        ret = self._entries.get(identifier)
        return False if ret is None else True

    def get[I: ServiceIdentifier[object]](
        self, identifier: Type[I]
    ) -> object | SyncDescriptor[I] | None:
        ret = self._entries.get(identifier)
        return ret
