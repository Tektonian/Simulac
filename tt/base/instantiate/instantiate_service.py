from inspect import signature
from typing import Generic, Iterable, Tuple, List, Type, Any, TypeVar, Type
from abc import ABC

from .descriptor import SyncDescriptor
from .instantiate import IInstantiateService, ServiceIdentifier, IServiceAccessor
from .service_collection import ServiceCollection


_ENABLE_TRACING = True

T = TypeVar("T")
G = Generic[T]
S = TypeVar("S", bound=ServiceIdentifier)


class InstantiateService(IInstantiateService):
    def __init__(
        self,
        _services: ServiceCollection = ServiceCollection([]),
        _strict: bool = False,
        _parent: IInstantiateService | None = None,
        _enable_tracing: bool = _ENABLE_TRACING,
    ):
        super().__init__()
        self._services = _services
        self._strict = _strict
        self._parent = _parent
        self._enable_tracing = _enable_tracing

        self._services.set(IInstantiateService, self)

    def _get_service_dependencies[I: ServiceIdentifier](
        self, descriptor: SyncDescriptor[I]
    ):
        sign = signature(descriptor.ctor)

        ret: List[Tuple[Type[ServiceIdentifier], int]] = []

        idx = 0
        for name, param in sign.parameters.items():
            if name == "self":
                continue
            annotation = param.annotation
            identifier = next(
                (
                    entry
                    for entry in self._services._entries
                    if entry.__name__ == annotation
                ),
                None,
            )
            if identifier is None:
                raise Exception(f"Unresolved service dependency: {annotation}")
            # for entry in self._services._entries:
            #     if entry.__name__ == annotation:
            #         identifier = entry
            ret.append((identifier, idx))
            idx += 1

        return ret

    # Lazy proxy: https://code.activestate.com/recipes/578014-lazy-load-object-proxying/
    # Proxy: https://web.archive.org/web/20220819152103/http://code.activestate.com/recipes/496741-object-proxying/

    def create_instance[I: ServiceIdentifier](
        self,
        descriptor: Type[SyncDescriptor[I]],
        *non_leading_service_args: tuple[Iterable[Any], ...],
    ):

        if isinstance(descriptor, SyncDescriptor):
            service_dependencies = self._get_service_dependencies(descriptor)
            print("service_dependencies", service_dependencies)
            service_args: List[
                ServiceIdentifier | SyncDescriptor[ServiceIdentifier]
            ] = []

            service_dependencies = sorted(
                service_dependencies, key=lambda dependency: dependency[1]
            )

            for dependency in service_dependencies:
                service_identifier, _ = dependency
                service = self._get_or_create_service_instance(service_identifier)
                print("dependency", service, service_identifier)
                service_args.append(service)

            args = descriptor.static_arguments + list(non_leading_service_args)
            first_service_arg_pos = (
                # Should be service_dependencies[0][1]?
                service_dependencies[0][1]
                if len(service_dependencies) > 0
                else len(args)
            )

            if len(args) != first_service_arg_pos:
                print(
                    "[createInstance] Service dependency error: ",
                    service_dependencies,
                    service_args,
                )

            final_args = args + service_args
            print(
                "[createInstance]",
                final_args,
                args,
                service_args,
                descriptor,
                descriptor.ctor,
                descriptor.static_arguments,
                # descriptor.ctor(),
            )
            instance = descriptor.ctor(*tuple(final_args))
            return instance

        else:
            if len(non_leading_service_args) == 0:
                return descriptor()
            else:
                raise Exception("Not implemented yet")

    def _set_created_service_instance[I: ServiceIdentifier](
        self, identifier: Type[I], instance: object
    ):
        if isinstance(self._services.get(identifier), SyncDescriptor):
            self._services.set(identifier, instance)

    def _get_or_create_service_instance[I: ServiceIdentifier](
        self, identifier: Type[I]
    ) -> I | SyncDescriptor[I]:
        ctorOrInstance = self._services.get(identifier)

        if isinstance(ctorOrInstance, SyncDescriptor):
            instance = self.create_instance(ctorOrInstance)
            self._set_created_service_instance(identifier, instance)
            return instance
        else:
            return ctorOrInstance

    @property
    def service_accessor(self) -> IServiceAccessor:

        service_self = self

        class ServiceAccessor(IServiceAccessor):

            def get(self, identifier: Type[T]) -> T:
                ret = service_self._get_or_create_service_instance(identifier)
                print("get", identifier, ret)
                return ret

        accessor = ServiceAccessor()

        return accessor
