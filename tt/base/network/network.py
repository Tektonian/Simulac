from enum import Enum


class SchemasEnum(Enum):
    """A schema that is used for models that exist in memory\
        only and that have no correspondence on a server or such.
    """

    IN_MEMORY = "inmemory"

    HTTP = "http"
    HTTPS = "https"
    FILE = "file"
