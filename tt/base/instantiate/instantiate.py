from __future__ import annotations
from inspect import signature
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from typing import TypeVar, Any, Type, Iterable, MutableMapping, Generic

if TYPE_CHECKING:
    from tt.base.instantiate.descriptor import SyncDescriptor

T = TypeVar("T")
I = TypeVar("I")


class IServiceAccessor(ABC):
    @abstractmethod
    def get(self, identifier: Type[I]) -> I:
        pass


"""
#region: TODO: Add service identifier decorator on head of Interface classes
See instantiate_service.py:InstantiateService._get_service_dependencies it is now using 'Type[ABC]' for service identification. This should be changed to _Utils.service_ids.get()
"""


A = TypeVar("A", bound=ABC)


class ServiceIdentifier(Generic[T]):
    pass


class _Util:
    service_ids: dict[str, type[ServiceIdentifier[Any]]] = {}


def service_identifier(identifier: str):

    def wrapper[T](cls: type[ServiceIdentifier[T]]) -> type[ServiceIdentifier[T]]:
        # sign = signature(cls)

        key = cls.__name__
        if identifier != key:
            raise BaseException("No maching key")
        _Util.service_ids[key] = cls
        return cls

    return wrapper


# endregion


@service_identifier("IInstantiateService")
class IInstantiateService(ServiceIdentifier[Any]):
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
