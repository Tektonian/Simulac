from abc import ABC, abstractmethod
from enum import Enum
from inspect import isclass, signature
from typing import Any, Generic, List, MutableMapping, Tuple, Type, TypeVar

from .descriptor import SyncDescriptor
from .instantiate import ServiceIdentifier


class InstantiateType(Enum):
    EAGER = True
    DELAYED = False


_registry: List[Tuple[Type[ServiceIdentifier[object]], SyncDescriptor[Any]]] = []


O = TypeVar("O", bound=object)


def register_singleton[O](
    interface: Type[ServiceIdentifier[O]],
    descriptor: Type[object],
    support_delayed_instantiation: InstantiateType = InstantiateType.DELAYED,
):
    if not issubclass(interface, ServiceIdentifier):
        raise Exception("")

    elif not isclass(descriptor):
        raise Exception("")

    elif not issubclass(descriptor, interface):
        raise Exception("")

    else:
        _registry.append(
            (
                interface,
                SyncDescriptor(descriptor, [], support_delayed_instantiation.value),
            )
        )


def get_singleton_service_descriptors():
    return _registry
