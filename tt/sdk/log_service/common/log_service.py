from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Tuple

import structlog

from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier


class LogLevel(Enum):
    OFF = 0
    TRACE = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    ERROR = 5


DEFAULT_LOG_LEVEL: LogLevel = LogLevel.INFO


@service_identifier("ILogService")
class ILogService(ServiceIdentifier["ILogService"]):
    @abstractmethod
    def set_level(self, level: LogLevel) -> None:
        pass

    @abstractmethod
    def get_level(self) -> LogLevel:
        pass

    @abstractmethod
    def trace(self, message: str, *args: Tuple[Any]) -> None:
        pass

    @abstractmethod
    def debug(self, message: str, *args: Tuple[Any]) -> None:
        pass

    @abstractmethod
    def info(self, message: str, *args: Tuple[Any]) -> None:
        pass

    @abstractmethod
    def warn(self, message: str, *args: Tuple[Any]) -> None:
        pass

    @abstractmethod
    def error(self, message: str, *args: Tuple[Any]) -> None:
        pass


class LogService(ILogService):
    def __init__(self):
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.level = DEFAULT_LOG_LEVEL

    def set_level(self, level: LogLevel):
        self.level = level

    def get_level(self):
        return self.level

    def trace(self, message: str, *args: tuple[Tuple[Any], ...]):
        self.logger.exception(message)

    def debug(self, message: str, *args: tuple[Tuple[Any], ...]):
        self.logger.debug(message, *args)

    def info(self, message: str, *args: tuple[Tuple[Any], ...]):
        self.logger.info(message, *args)

    def warn(self, message: str, *args: tuple[Tuple[Any], ...]):
        self.logger.warn(message, *args)

    def error(self, message: str, *args: tuple[Tuple[Any], ...]):
        self.logger.error(message, *args)
