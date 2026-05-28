from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import SplitResult

from simulac.sdk.file_service.common.files import (
    FileTypeEnum,
    IFileSystemProvider,
    IFileWriteOption,
    IStat,
)


class DiskFileSystemProvider(IFileSystemProvider):
    def _path(self, uri: SplitResult) -> Path:
        return Path(uri.path)

    def stat(self, uri: SplitResult):
        try:
            path = self._path(uri)
            st = path.lstat()
            if path.is_symlink():
                kind = FileTypeEnum.SYMBOLICLINK
            elif path.is_dir():
                kind = FileTypeEnum.DIRECTORY
            elif path.is_file():
                kind = FileTypeEnum.FILE
            else:
                kind = FileTypeEnum.UNKNOWN
            return IStat(kind, st.st_size), None
        except BaseException as exc:
            return None, exc

    def mkdir(self, uri: SplitResult):
        try:
            self._path(uri).mkdir(parents=True, exist_ok=True)
            return True, None
        except BaseException as exc:
            return None, exc

    def readdir(self, uri: SplitResult):
        try:
            rows: list[tuple[str, FileTypeEnum]] = []
            for child in self._path(uri).iterdir():
                if child.is_dir():
                    kind = FileTypeEnum.DIRECTORY
                elif child.is_file():
                    kind = FileTypeEnum.FILE
                elif child.is_symlink():
                    kind = FileTypeEnum.SYMBOLICLINK
                else:
                    kind = FileTypeEnum.UNKNOWN
                rows.append((child.name, kind))
            return IStat(FileTypeEnum.DIRECTORY, 0), rows
        except BaseException as exc:
            return None, exc

    def delete(self, uri: SplitResult, recursive: bool):
        try:
            path = self._path(uri)
            if path.is_dir() and recursive:
                shutil.rmtree(path)
            elif path.is_dir():
                path.rmdir()
            else:
                path.unlink(missing_ok=True)
            return True, None
        except BaseException as exc:
            return None, exc

    def rename(self, from_file: SplitResult, to_file: SplitResult):
        try:
            self._path(from_file).rename(self._path(to_file))
            return True, None
        except BaseException as exc:
            return None, exc

    def copy(self, from_file: SplitResult, to_file: SplitResult):
        try:
            shutil.copy2(self._path(from_file), self._path(to_file))
            return True, None
        except BaseException as exc:
            return None, exc

    def read_file(self, uri: SplitResult):
        try:
            return self._path(uri).read_bytes(), None
        except BaseException as exc:
            return None, exc

    def write_file(self, uri: SplitResult, data: bytes, opts: IFileWriteOption):
        try:
            path = self._path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if opts.append else "wb"
            if path.exists() and not opts.overwrite and not opts.append:
                raise FileExistsError(str(path))
            with path.open(mode) as file:
                file.write(data)
            return True, None
        except BaseException as exc:
            return None, exc

    def clone_file(self, from_file: SplitResult, to_file: SplitResult):
        return self.copy(from_file, to_file)
