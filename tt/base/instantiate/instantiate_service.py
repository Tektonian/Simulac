from __future__ import annotations

import inspect
import time
import traceback
from enum import IntEnum
from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from .descriptor import SyncDescriptor
from .graph import Graph
from .instantiate import IInstantiateService, IServiceAccessor, ServiceIdentifier, _Util
from .service_collection import ServiceCollection

_ENABLE_TRACING = False

T = TypeVar("T")
G = Generic[T]


class CyclicDependencyError(BaseException):
    def __init__(self, graph: Graph[Any]) -> None:
        message = graph.find_cycle_slow()
        message = (
            f"UNABLE to detect cycle, dumping graph: \n${graph.to_string()}"
            if message is None
            else message
        )
        self.message = message
        super().__init__("cyclic dependency between services \n" + message)


class InstantiateService(IInstantiateService):
    _global_graph: Graph[str] | None = None
    _global_graph_implicit_dependency: str | None = None
    _children: set[InstantiateService] = set()

    def __init__(
        self,
        _services: ServiceCollection = ServiceCollection([]),
        _strict: bool = False,
        _parent: InstantiateService | None = None,
        _enable_tracing: bool = _ENABLE_TRACING,
    ):
        super().__init__()
        self._services = _services
        self._strict = _strict
        self._parent = _parent
        self._enable_tracing = _enable_tracing

        self._services.set(IInstantiateService, self)

        if _ENABLE_TRACING:
            if _parent is not None and _parent._global_graph is not None:
                self._global_graph = _parent._global_graph
            else:
                self._global_graph = Graph[str](lambda d: d)

    def _get_service_dependencies[I: object](self, ctor: Type[I]):

        sign = inspect.signature(ctor)
        ret: List[Tuple[Type[ServiceIdentifier[object]], int]] = []
        idx = 0

        for name, param in sign.parameters.items():
            if name == "self":
                continue
            annotation: str | inspect.Parameter.empty = param.annotation
            identifier: Type[ServiceIdentifier[Any]] | None = None

            if isinstance(annotation, str):
                identifier = _Util.service_ids.get(annotation)
            elif inspect.isclass(annotation) and issubclass(
                annotation, ServiceIdentifier
            ):
                # Won't be reached maybe?
                identifier = annotation

            if identifier is None:
                raise Exception(f"Unresolved service dependency: {annotation}")
            else:
                ret.append((identifier, idx))
            idx += 1

        return ret

    # Lazy proxy: https://code.activestate.com/recipes/578014-lazy-load-object-proxying/
    # Proxy: https://web.archive.org/web/20220819152103/http://code.activestate.com/recipes/496741-object-proxying/

    def create_instance[I: ServiceIdentifier[object]](
        self,
        ctor_or_descriptor: Type[object] | Type[SyncDescriptor[I]],
        *non_leading_service_args: tuple[Iterable[Any], ...],
    ) -> I:

        _trace: Trace
        result: Any

        if isinstance(ctor_or_descriptor, SyncDescriptor):
            _trace = Trace.trace_creation(self._enable_tracing, ctor_or_descriptor.ctor)
            result = self._create_instance(
                ctor_or_descriptor.ctor,
                ctor_or_descriptor.static_arguments + list(non_leading_service_args),
                _trace,
            )
        else:
            _trace = Trace.trace_creation(self._enable_tracing, ctor_or_descriptor)
            result = self._create_instance(
                ctor_or_descriptor, list(non_leading_service_args), _trace
            )

        _trace.stop()

        return result

    def _create_instance[T: object](
        self, ctor: Type[T], args: list[Any], _trace: Trace
    ) -> T:
        service_dependencies = self._get_service_dependencies(ctor)
        service_args: List[
            ServiceIdentifier[Any] | SyncDescriptor[ServiceIdentifier[Any]] | Any
        ] = []
        service_dependencies = sorted(
            service_dependencies, key=lambda dependency: dependency[1]
        )

        for dependency in service_dependencies:
            service_identifier, _ = dependency
            service = self._get_or_create_service_instance(service_identifier, _trace)
            if service is None:
                raise BaseException(
                    f"[createInstance] {ctor} depends on UNKNOWN service {dependency[0]}."
                )
            service_args.append(service)

        if hasattr(ctor, "static_arguments"):
            args = ctor.static_arguments + args

        first_service_arg_pos = (
            # Should be service_dependencies[0][1]?
            service_dependencies[0][1] if len(service_dependencies) > 0 else len(args)
        )

        if len(args) != first_service_arg_pos:
            stack_summary = traceback.extract_stack(None, 5)
            print(
                f"[createInstance] First service dependency of ${ctor.name} at position {first_service_arg_pos + 1} conflicts with {len(args)} static arguments",
                service_dependencies,
                service_args,
                stack_summary,
            )
            delta = first_service_arg_pos - len(args)
            if delta > 0:
                args = args + [None for _ in range(delta)]
            else:
                args = args[0:first_service_arg_pos]

        final_args = args + service_args

        # TODO: need to be proxied
        instance = ctor(*tuple(final_args))
        return instance

    def _set_created_service_instance[I: ServiceIdentifier[Any]](
        self, identifier: Type[I], instance: object
    ):
        if isinstance(self._services.get(identifier), SyncDescriptor):
            self._services.set(identifier, instance)

    def _get_or_create_service_instance(
        self,
        identifier: Type[ServiceIdentifier[object]],
        _trace: Trace,
    ) -> object:

        if self._global_graph and self._global_graph_implicit_dependency is not None:
            self._global_graph.insert_edges(
                self._global_graph_implicit_dependency, str(identifier)
            )

        ctor_or_instance = self._get_service_instance_or_descriptor(identifier)

        if isinstance(ctor_or_instance, SyncDescriptor):
            # TODO: Breaking change here. Need Test!
            # instance = self.create_instance(ctor_or_instance)
            # self._set_created_service_instance(identifier, instance)
            # return instance
            instance = self._safe_create_and_cache_service_instance(
                identifier, ctor_or_instance, _trace.branch(identifier, True)
            )
            return instance
        else:
            _trace.branch(identifier, False)
            return ctor_or_instance

    def _get_service_instance_or_descriptor(
        self, identifier: Type[ServiceIdentifier[object]]
    ) -> SyncDescriptor[ServiceIdentifier[object]] | object | None:
        instance_or_desc = self._services.get(identifier)
        if instance_or_desc is None and self._parent is not None:
            return self._parent._get_service_instance_or_descriptor(identifier)
        else:
            return instance_or_desc

    _active_instantiations: set[Type[ServiceIdentifier[Any]]] = set()

    def _safe_create_and_cache_service_instance[O: object](
        self,
        identifier: Type[ServiceIdentifier[O]],
        desc: SyncDescriptor[ServiceIdentifier[O]],
        _trace: Trace,
    ) -> O:
        if identifier in self._active_instantiations:
            raise BaseException(
                f"illegal state - RECURSIVELY instantiating service '{identifier}'"
            )
        self._active_instantiations.add(identifier)
        try:
            return self._create_and_cache_service_instance(identifier, desc, _trace)
        finally:
            self._active_instantiations.remove(identifier)

    def _create_and_cache_service_instance[I: object](
        self,
        identifier: Type[ServiceIdentifier[I]],
        desc: SyncDescriptor[ServiceIdentifier[I]],
        _trace: Trace,
    ) -> I:
        graph = Graph[Tuple[Type[ServiceIdentifier[Any]], SyncDescriptor[Any], Trace]](
            lambda d: str(d[0])
        )
        cycle_count = 0
        stack: list[Tuple[Type[ServiceIdentifier[Any]], SyncDescriptor[Any], Trace]] = [
            (identifier, desc, _trace)
        ]
        seen: set[str] = set()

        while len(stack) > 0:
            item = stack.pop()

            if str(item[0]) in seen:
                continue

            seen.add(str(item[0]))
            graph.lookup_or_insert_node(item)

            # a weak but working heuristic for cycle checks
            cycle_count += 1
            if cycle_count > 1000:
                raise CyclicDependencyError(graph)

            # check all dependencies for existence and if they need to be created first
            dependencies = self._get_service_dependencies(item[1].ctor)
            for dependency in dependencies:
                instance_or_desc = self._get_service_instance_or_descriptor(
                    dependency[0]
                )
                if instance_or_desc is None:
                    raise BaseException(
                        f"[createInstance] ${identifier} depends on ${dependency[0]} which is NOT registered."
                    )

                if self._global_graph is not None:
                    self._global_graph.insert_edges(str(item[0]), str(dependency[0]))

                if isinstance(instance_or_desc, SyncDescriptor):
                    d: Tuple[
                        Type[ServiceIdentifier[Any]], SyncDescriptor[Any], Trace
                    ] = (
                        dependency[0],
                        instance_or_desc,
                        item[2].branch(dependency[0], True),
                    )
                    graph.insert_edges(item, d)
                    stack.append(d)

        while True:
            roots = graph.roots()

            # if there is no more roots but still
            # nodes in the graph we have a cycle
            if len(roots) == 0:
                if graph.is_empty() is False:
                    raise CyclicDependencyError(graph)
                break

            for node in roots:
                data = node.data
                instance_or_desc = self._get_service_instance_or_descriptor(data[0])

                if isinstance(instance_or_desc, SyncDescriptor):
                    # TODO: https://github.com/microsoft/vscode/blob/1aa235610cf3a1cea299d2c8a5f1bc80f33027c1/src/vs/platform/instantiation/common/instantiationService.ts#L282
                    # raise BaseException("not implemented yet")
                    instance = self._create_instance(
                        instance_or_desc.ctor,
                        instance_or_desc.static_arguments,
                        data[2],
                    )
                    self._set_created_service_instance(data[0], instance)
                graph.remove_node(data)

        return self._get_service_instance_or_descriptor(identifier)

    @property
    def service_accessor(self) -> IServiceAccessor:

        service_self = self

        _done = False

        class ServiceAccessor(IServiceAccessor):
            def get[O: object](self, identifier: type[ServiceIdentifier[O]]) -> O:
                nonlocal _done

                __obj = {"name": "service_accessor"}
                _trace = Trace.trace_invocation(service_self._enable_tracing, __obj)

                if _done:
                    raise BaseException(
                        "service accessor is only valid during the invocation of its target method"
                    )

                ret = service_self._get_or_create_service_instance(identifier, _trace)

                if ret is None and service_self._strict:
                    raise BaseException(
                        f"[invokeFunction] unknown service '{identifier}'"
                    )

                _trace.stop()
                return cast(T, ret)

        accessor = ServiceAccessor()

        return accessor


class TraceTypeEnum(IntEnum):
    NONE = 0
    CREATION = 1
    INVORATION = 2
    BRANCH = 3


class Trace:
    all = set[str]()

    @staticmethod
    def trace_invocation(_enable_tracing: bool, ctor: object) -> Trace:
        if hasattr(ctor, "name") is True:
            ctor: str = ctor["name"]
        else:
            # TODO: https://github.com/microsoft/vscode/blob/1aa235610cf3a1cea299d2c8a5f1bc80f33027c1/src/vs/platform/instantiation/common/instantiationService.ts#L417
            ctor = str(ctor)

        if _enable_tracing is True:
            return Trace(TraceTypeEnum.INVORATION, ctor)
        else:
            return _NoneTrace()

    @staticmethod
    def trace_creation(_enable_tracing: bool, ctor: object) -> Trace:
        if hasattr(ctor, "name") is True:
            ctor = ctor.name
        else:
            # TODO: https://github.com/microsoft/vscode/blob/1aa235610cf3a1cea299d2c8a5f1bc80f33027c1/src/vs/platform/instantiation/common/instantiationService.ts#L421
            ctor = str(ctor)
        if _enable_tracing is True:
            return Trace(TraceTypeEnum.CREATION, ctor)
        else:
            return _NoneTrace()

    _totals = 0

    def __init__(self, type: TraceTypeEnum, name: str | None) -> None:
        self.type = type
        self.name = name
        self._start = time.time()
        self._dep: list[Tuple[Type[ServiceIdentifier[Any]], bool, Optional[Trace]]] = []

    def branch(self, identifier: Type[ServiceIdentifier[Any]], first: bool) -> Trace:
        child = Trace(TraceTypeEnum.BRANCH, str(identifier))
        self._dep.append((identifier, first, child))
        return child

    def stop(self):
        dur = time.time() - self._start
        Trace._totals += dur

        caused_creation = False

        def print_child(n: int, trace: Trace) -> None | str:
            nonlocal caused_creation

            res: list[str] = []
            prefix = "\t".join(["" for _ in range(n + 1)])
            for identifier, first, child in trace._dep:
                if first is True and child is not None:
                    caused_creation = True
                    res.append(f"{prefix}CREATES -> {identifier}")
                    nested = print_child(n + 1, child)
                    if nested is not None:
                        res.append(nested)
                else:
                    res.append(f"{prefix}uses -> {identifier}")
            return "\n".join(res)

        lines = [
            f"{'CREATE' if self.type == TraceTypeEnum.CREATION else 'CALL'} {self.name}",
            f"{print_child(1, self)}",
            f"DONE, took {dur.__format__('.4f')}ms (grand total {Trace._totals.__format__('.4f')}ms)",
        ]
        if (dur > 2 or caused_creation) or _ENABLE_TRACING:
            Trace.all.add("\n".join(lines))
            print("\n".join(lines))


class _NoneTrace(Trace):
    def __init__(self) -> None:
        pass

    def branch(self, identifier: Type[ServiceIdentifier[Any]], first: bool):
        return self

    def stop(self):
        return
