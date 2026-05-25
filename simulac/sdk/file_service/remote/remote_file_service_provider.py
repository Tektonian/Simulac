from __future__ import annotations

import urllib.request
from urllib.parse import SplitResult, urlunsplit

from simulac.sdk.file_service.common.files import (
    FileTypeEnum,
    IFileSystemProvider,
    IFileWriteOption,
    IStat,
)


class HttpFileSystemProvider(IFileSystemProvider):
    _HEADERS = {
        "User-Agent": "simulac-python-sdk",
        "Accept": "*/*",
    }

    def _url(self, uri: SplitResult) -> str:
        return urlunsplit(uri)

    def _request(self, uri: SplitResult, *, method: str = "GET") -> urllib.request.Request:
        return urllib.request.Request(
            self._url(uri),
            headers=self._HEADERS,
            method=method,
        )

    def stat(self, uri: SplitResult):
        try:
            req = self._request(uri, method="HEAD")
            with urllib.request.urlopen(req) as res:
                size = int(res.headers.get("content-length") or 0)
            return IStat(FileTypeEnum.FILE, size), None
        except BaseException as exc:
            return None, exc

    def read_file(self, uri: SplitResult):
        try:
            with urllib.request.urlopen(self._request(uri)) as res:
                return res.read(), None
        except BaseException as exc:
            return None, exc

    def mkdir(self, uri):
        return None, NotImplementedError("http mkdir is unsupported")

    def readdir(self, uri):
        return None, NotImplementedError("http readdir is unsupported")

    def delete(self, uri, recursive):
        return None, NotImplementedError("http delete is unsupported")

    def rename(self, from_file, to_file):
        return None, NotImplementedError("http rename is unsupported")

    def copy(self, from_file, to_file):
        return None, NotImplementedError("http copy is unsupported")

    def write_file(self, uri, data: bytes, opts: IFileWriteOption):
        return None, NotImplementedError("http write is unsupported")

    def clone_file(self, from_file, to_file):
        return None, NotImplementedError("http clone is unsupported")
