from abc import abstractmethod
from pathlib import Path

from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier


@service_identifier("IEnvvarService")
class IEnvvarService(ServiceIdentifier["IEnvvarService"]):
    # logging
    @property
    @abstractmethod
    def log_level(self) -> str: ...

    # telemetry
    @property
    @abstractmethod
    def telemetry_disabled(self) -> bool: ...

    # file path
    @property
    @abstractmethod
    def log_file(self) -> Path: ...

    @property
    @abstractmethod
    def app_root(self) -> Path:
        """Root path, where this package is being used"""

    @property
    @abstractmethod
    def user_home(self) -> Path: ...

    @property
    @abstractmethod
    def tmp_dir(self) -> Path: ...

    @property
    @abstractmethod
    def cache_dir(self) -> Path: ...

    @property
    @abstractmethod
    def asset_dir(self) -> Path: ...

    @property
    @abstractmethod
    def token_path(self) -> Path: ...

    # token
    @property
    @abstractmethod
    def token(self) -> str | None: ...

    # url
    @property
    @abstractmethod
    def base_url(self) -> str: ...

    # develop
    @property
    @abstractmethod
    def dev_mode(self) -> bool: ...
