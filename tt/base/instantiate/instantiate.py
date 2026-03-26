from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    MutableMapping,
    Type,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from tt.base.instantiate.descriptor import SyncDescriptor

T = TypeVar("T")
I = TypeVar("I")


class IServiceAccessor(ABC):
    @abstractmethod
    def get[O: object](self, identifier: type[ServiceIdentifier[O]]) -> O:
        pass


"""
#region: TODO: Add service identifier decorator on head of Interface classes
See instantiate_service.py:InstantiateService._get_service_dependencies it is now using 'Type[ABC]' for service identification. This should be changed to _Utils.service_ids.get()
"""


A = TypeVar("A", bound=ABC)


class ServiceIdentifier(Generic[T]):
    pass


class _Util:
    service_ids: dict[str, Type[ServiceIdentifier[Any]]] = {}


def service_identifier(identifier: str):

    def wrapper[T](cls: Type[ServiceIdentifier[T]]) -> Type[T]:
        # sign = signature(cls)

        key = cls.__name__
        if identifier != key:
            raise BaseException("No maching key")
        _Util.service_ids[key] = cls
        return cast(Type[T], cls)

    return wrapper


# endregion


@service_identifier("IInstantiateService")
class IInstantiateService(ServiceIdentifier["IInstantiateService"]):
    @abstractmethod
    def create_instance[I: ServiceIdentifier[Any]](
        self,
        ctor_or_descriptor: Type[object] | Type[SyncDescriptor[I]],
        *non_leading_service_args: tuple[Iterable[Any], ...],
    ) -> I:
        pass

    @property
    @abstractmethod
    def service_accessor(self) -> IServiceAccessor:
        pass
