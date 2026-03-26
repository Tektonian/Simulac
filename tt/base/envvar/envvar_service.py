import os
import sys
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from tt.base.envvar.envvar import IEnvvarService

if TYPE_CHECKING:
    from tt.sdk.log_service.common.log_service import LogLevel


class EnvvarKeyValue(StrEnum):
    TT_LOG_LEVEL = "TT_LOG_LEVEL"
    TT_BASE_URL = "TT_BASE_URL"
    TT_TELEMETRY = "TT_TELEMETRY"
    TT_API_KEY = "TT_API_KEY"


class EnvvarService(IEnvvarService):
    BASE_URL = "https://tektonian.com/api"

    KEY_VALUE = EnvvarKeyValue

    def __init__(self) -> None:
        self._log_level = "info"

        self._user_home = os.path.expanduser("~")
        self._tmp_dir = tempfile.gettempdir()
        self._app_root = os.getcwd()

    @property
    def log_level(self) -> str:
        level = os.environ.get(self.KEY_VALUE.TT_LOG_LEVEL, None)
        if level in ["off", "trace", "debug", "info", "warning", "error"]:
            return level
        else:
            return self._log_level

    @property
    def telemetry_disabled(self) -> bool:
        tele_env = os.environ.get(self.KEY_VALUE.TT_TELEMETRY, None)
        if tele_env is None:
            return False
        elif tele_env == "off":
            return True

        return False

    @property
    def log_file(self) -> Path:
        tt_dir = os.path.join(self.app_root, ".tt", "log.json")
        return Path(tt_dir)

    @property
    def app_root(self) -> Path:
        return Path(self._app_root)

    @property
    def user_home(self) -> Path:
        return Path(self._user_home)

    @property
    def tmp_dir(self) -> Path:
        return Path(self._tmp_dir)

    @property
    def cache_dir(self) -> Path:
        return Path(os.path.join(self.user_home, ".cache"))

    @property
    def asset_dir(self) -> Path:
        return Path(os.path.join(self.cache_dir, "asset"))

    @property
    def token_path(self) -> Path:
        return Path(os.path.join(self.cache_dir, "token"))

    @property
    def token(self) -> str | None:
        token_env = os.environ.get(self.KEY_VALUE.TT_API_KEY.value, None)
        if token_env is not None:
            return token_env

        token_ret = ""
        if self.token_path.exists() and self.token_path.is_file():
            with open(self.token_path, "r") as file:
                token_ret = file.readline(1)

        token_ret = token_ret.replace("\n", "").replace("\r", "").strip()

        if token_ret.startswith("tt_") and len(token_ret) > 40:
            return token_ret

        return None

    @property
    def base_url(self) -> str:
        env_path = os.environ.get(self.KEY_VALUE.TT_BASE_URL.value, None)
        if env_path is not None:
            return env_path
        else:
            return self.BASE_URL

    @property
    def dev_mode(self) -> bool:
        _dev_mode = os.environ.get("PYTHONDEVMODE", False)

        if _dev_mode is False:
            _dev_mode = True if sys.warnoptions == ["default"] else False
        if isinstance(_dev_mode, str):
            _dev_mode = True if _dev_mode == "1" else False
        return _dev_mode
