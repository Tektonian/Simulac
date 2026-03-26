from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, List, Optional, Tuple

from tt.base.error.error import TektonianBaseError
from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from tt.base.result.result import ResultType

if TYPE_CHECKING:
    import urllib.parse


class FileTypeEnum(IntEnum):
    UNKNOWN = 0
    FILE = 1
    DIRECTORY = 2
    SYMBOLICLINK = 64


@dataclass(frozen=True)
class IStat(ABC):
    type: FileTypeEnum
    size: int


@dataclass(frozen=True)
class IFileWriteOption:
    overwrite: bool
    create: bool
    append: bool


class IFileSystemProvider(ABC):
    @abstractmethod
    def stat(
        self, uri: urllib.parse.SplitResult
    ) -> ResultType[IStat, BaseException]: ...

    @abstractmethod
    def mkdir(
        self, uri: urllib.parse.SplitResult
    ) -> ResultType[bool, BaseException]: ...

    @abstractmethod
    def readdir(
        self, uri: urllib.parse.SplitResult
    ) -> ResultType[IStat, List[Tuple[str, FileTypeEnum]]]: ...

    @abstractmethod
    def delete(
        self, uri: urllib.parse.SplitResult, recursive: bool
    ) -> ResultType[bool, BaseException]: ...

    @abstractmethod
    def rename(
        self, from_file: urllib.parse.SplitResult, to_file: urllib.parse.SplitResult
    ) -> ResultType[bool, BaseException]: ...

    @abstractmethod
    def copy(
        self, from_file: urllib.parse.SplitResult, to_file: urllib.parse.SplitResult
    ) -> ResultType[bool, BaseException]: ...

    @abstractmethod
    def read_file(
        self,
        uri: urllib.parse.SplitResult,
    ) -> ResultType[bytes, BaseException]: ...

    @abstractmethod
    def write_file(
        self, uri: urllib.parse.SplitResult, opts: IFileWriteOption
    ) -> ResultType[bool, BaseException]: ...

    @abstractmethod
    def clone_file(
        self, from_file: urllib.parse.SplitResult, to_file: urllib.parse.SplitResult
    ) -> ResultType[bool, BaseException]: ...


@dataclass(frozen=True)
class IFileStat:
    resource: urllib.parse.SplitResult
    name: str
    is_file: bool
    is_directory: bool
    is_symbolic: bool
    children: List[IFileStat]


@dataclass(frozen=True)
class IResolveFileOptions:
    """Automatically continue resolving children of a directory until the provided resources
    are found.
    """

    resolve_to: List[urllib.parse.SplitResult] | None


@service_identifier("IFileService")
class IFileService(ServiceIdentifier["IFileService"]):
    @abstractmethod
    def register_provider(self, schema: str, provider: IFileSystemProvider) -> None: ...

    @abstractmethod
    def get_provider(self, schema: str) -> IFileSystemProvider | None: ...

    @abstractmethod
    def has_provider(self, resource: urllib.parse.SplitResult) -> bool: ...

    @abstractmethod
    def resolve(
        self, resource: urllib.parse.SplitResult, options: IResolveFileOptions
    ) -> ResultType[IFileStat, BaseException]: ...

    @abstractmethod
    def resolve_bulk(
        self, to_resolve: List[Tuple[urllib.parse.SplitResult, IResolveFileOptions]]
    ) -> ResultType[List[IFileStat], BaseException]: ...

    @abstractmethod
    def stat(
        self, resource: urllib.parse.SplitResult
    ) -> ResultType[IFileStat, BaseException]: ...

    @abstractmethod
    def find_real_path(
        self, resource: urllib.parse.SplitResult
    ) -> ResultType[urllib.parse.SplitResult, BaseException]:
        """real path can be different if resource uri is symbolic link"""

    @abstractmethod
    def exists(self, resource: urllib.parse.SplitResult) -> bool: ...

    @abstractmethod
    def read_file(
        self, resource: urllib.parse.SplitResult
    ) -> ResultType[bytes, BaseException]: ...

    @abstractmethod
    def write_file(
        self,
        resource: urllib.parse.SplitResult,
        data: bytes,
        append: bool | None = False,
    ) -> ResultType[IFileStat, BaseException]: ...

    @abstractmethod
    def copy(
        self,
        resource: urllib.parse.SplitResult,
        target: urllib.parse.SplitResult,
        overwrite: bool | None,
    ) -> ResultType[IFileStat, BaseException]: ...

    @abstractmethod
    def create_file(
        self,
        resource: urllib.parse.SplitResult,
        data: bytes,
        overwrite: bool | None = True,
    ) -> ResultType[IFileStat, BaseException]: ...

    @abstractmethod
    def create_folder(
        self, resource: urllib.parse.SplitResult
    ) -> ResultType[IFileStat, BaseException]: ...

    @abstractmethod
    def delete(
        self, resource: urllib.parse.SplitResult, recursive: bool
    ) -> ResultType[bool, BaseException]: ...


class FileSystemProviderErrorCode(Enum):
    FILE_EXSITS = "EntryExists"
    FILE_NOT_FOUND = "EntryNotFound"
    ENTRY_NOT_DIRECTORY = "EntryNotADirectory"
    ENTRY_IS_DIRECTORY = "EntryIsADirectory"
    ENTRY_EXCEED_STORAGE_QUATA = "EntryExceedsStorageQuota"
    FILE_TOO_LARGE = "EntryTooLarge"
    FILE_WRITE_LOCKED = "EntryWriteLocked"
    NO_PERMISSION = "NoPermissions"
    UNAVAILABLE = "Unavailable"
    UNKONWN = "Unknown"


class FileSystemProviderError(TektonianBaseError):
    def __init__(self, message: str, code: FileSystemProviderErrorCode) -> None:
        super().__init__(message, dict())
        self.code = code
        self.name = f"{code.value} (FileSystemError)"

    @staticmethod
    def create(
        error: Exception | str, code: FileSystemProviderErrorCode
    ) -> "FileSystemProviderError":
        message = str(error)
        provider_error = FileSystemProviderError(message, code)
        mark_as_file_system_provider_error(provider_error, code)
        return provider_error


def create_file_system_provider_error(
    error: Exception | str, code: FileSystemProviderErrorCode
) -> FileSystemProviderError:
    return FileSystemProviderError.create(error, code)


def ensure_file_system_provider_error(
    error: Optional[BaseException] = None,
) -> BaseException:
    if not error:
        return create_file_system_provider_error(
            "Unknown Error", FileSystemProviderErrorCode.UNKONWN
        )
    return error


def mark_as_file_system_provider_error(
    error: BaseException, code: FileSystemProviderErrorCode
) -> BaseException:
    setattr(error, "name", f"{code.value} (FileSystemError)")
    if not hasattr(error, "code"):
        try:
            setattr(error, "code", code)  # may fail for some built-ins; safe to ignore
        except Exception:
            pass
    return error


_FILE_SYSTEM_ERROR_NAME_RE = re.compile(r"^(.+)\s+\(FileSystemError\)$")


def to_file_system_provider_error_code(
    error: Optional[BaseException],
) -> FileSystemProviderErrorCode:
    if not error:
        return FileSystemProviderErrorCode.UNKONWN

    if isinstance(error, FileSystemProviderError):
        return error.code

    name = getattr(error, "name", type(error).__name__)
    m = _FILE_SYSTEM_ERROR_NAME_RE.match(name)
    if not m:
        return FileSystemProviderErrorCode.UNKONWN

    mapped = m.group(1)
    # Map back to enum by value (TS stores the string value in name)
    for code in FileSystemProviderErrorCode:
        if code.value == mapped:
            return code

    return FileSystemProviderErrorCode.UNKONWN


class FileOperationResult(Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_IS_DIRECTORY = "FILE_IS_DIRECTORY"
    FILE_NOT_DIRECTORY = "FILE_NOT_DIRECTORY"
    FILE_WRITE_LOCKED = "FILE_WRITE_LOCKED"
    FILE_PERMISSION_DENIED = "FILE_PERMISSION_DENIED"
    FILE_MOVE_CONFLICT = "FILE_MOVE_CONFLICT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_OTHER_ERROR = "FILE_OTHER_ERROR"


class FileOperationError(TektonianBaseError):
    def __init__(
        self, message: str, file_operation_result: FileOperationResult
    ) -> None:
        super().__init__(message, {})
        self.file_operation_result = file_operation_result


def to_file_operation_result(error: BaseException) -> FileOperationResult:
    if isinstance(error, FileOperationError):
        return error.file_operation_result

    code = to_file_system_provider_error_code(error)

    if code == FileSystemProviderErrorCode.FILE_NOT_FOUND:
        return FileOperationResult.FILE_NOT_FOUND
    if code == FileSystemProviderErrorCode.ENTRY_IS_DIRECTORY:
        return FileOperationResult.FILE_IS_DIRECTORY
    if code == FileSystemProviderErrorCode.ENTRY_NOT_DIRECTORY:
        return FileOperationResult.FILE_NOT_DIRECTORY
    if code == FileSystemProviderErrorCode.FILE_WRITE_LOCKED:
        return FileOperationResult.FILE_WRITE_LOCKED
    if code == FileSystemProviderErrorCode.NO_PERMISSION:
        return FileOperationResult.FILE_PERMISSION_DENIED
    if code == FileSystemProviderErrorCode.FILE_EXSITS:
        return FileOperationResult.FILE_MOVE_CONFLICT
    if code == FileSystemProviderErrorCode.FILE_TOO_LARGE:
        return FileOperationResult.FILE_TOO_LARGE

    return FileOperationResult.FILE_OTHER_ERROR
