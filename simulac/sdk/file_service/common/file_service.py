from __future__ import annotations

from urllib.parse import SplitResult

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.file_service.common.files import (
    IFileService,
    IFileStat,
    IFileSystemProvider,
    IFileWriteOption,
    IResolveFileOptions,
)
from simulac.sdk.file_service.local.disk_file_service_provider import (
    DiskFileSystemProvider,
)
from simulac.sdk.file_service.remote.remote_file_service_provider import (
    HttpFileSystemProvider,
)


class FileService(IFileService):
    def __init__(self) -> None:
        self._providers: dict[str, IFileSystemProvider] = {}
        self.register_provider("file", DiskFileSystemProvider())
        self.register_provider("http", HttpFileSystemProvider())
        self.register_provider("https", HttpFileSystemProvider())

    def register_provider(self, schema: str, provider: IFileSystemProvider) -> None:
        self._providers[schema] = provider

    def get_provider(self, schema: str) -> IFileSystemProvider | None:
        return self._providers.get(schema)

    def has_provider(self, resource: SplitResult) -> bool:
        return resource.scheme in self._providers

    def _provider(self, resource: SplitResult) -> IFileSystemProvider:
        provider = self.get_provider(resource.scheme)
        if provider is None:
            raise SimulacBaseError(f"No file provider for scheme: {resource.scheme!r}")
        return provider

    def stat(self, resource: SplitResult):
        provider = self._provider(resource)
        stat, err = provider.stat(resource)
        if err:
            return None, err
        name = resource.path.rstrip("/").split("/")[-1]
        return IFileStat(
            resource=resource,
            name=name,
            is_file=stat.type == 1,
            is_directory=stat.type == 2,
            is_symbolic=stat.type == 64,
            children=[],
        ), None

    def exists(self, resource: SplitResult) -> bool:
        _, err = self.stat(resource)
        return err is None

    def read_file(self, resource: SplitResult):
        return self._provider(resource).read_file(resource)

    def write_file(
        self, resource: SplitResult, data: bytes, append: bool | None = False
    ):
        provider = self._provider(resource)
        ok, err = provider.write_file(
            resource,
            data,
            IFileWriteOption(overwrite=True, create=True, append=bool(append)),
        )
        if err:
            return None, err
        return self.stat(resource)

    def create_file(
        self, resource: SplitResult, data: bytes, overwrite: bool | None = True
    ):
        provider = self._provider(resource)
        ok, err = provider.write_file(
            resource,
            data,
            IFileWriteOption(overwrite=bool(overwrite), create=True, append=False),
        )
        if err:
            return None, err
        return self.stat(resource)

    def create_folder(self, resource: SplitResult):
        ok, err = self._provider(resource).mkdir(resource)
        if err:
            return None, err
        return self.stat(resource)

    def copy(self, resource: SplitResult, target: SplitResult, overwrite: bool | None):
        data, err = self.read_file(resource)
        if err:
            return None, err
        return self.create_file(target, data, overwrite=overwrite)

    def delete(self, resource: SplitResult, recursive: bool):
        return self._provider(resource).delete(resource, recursive)

    def resolve(self, resource: SplitResult, options: IResolveFileOptions):
        return self.stat(resource)

    def resolve_bulk(self, to_resolve):
        out = []
        for resource, options in to_resolve:
            stat, err = self.resolve(resource, options)
            if err:
                return None, err
            out.append(stat)
        return out, None

    def find_real_path(self, resource: SplitResult):
        return resource, None
