import typing
from typing import Union

import cv2.typing
import duckdb
import duckdb.sqltypes as sqltypes
import numpy


class Range1D:
    """A simple 1D range type.

    This implementation serializes to/from JSON so it can be stored in DuckDB
    as a VARCHAR. If you need a native DuckDB mapping, implement the
    DuckDBPyType interface per duckdb docs and register the type with duckdb.
    """

    def __init__(self, start: Union[int, float], end: Union[int, float]):
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"Range1D(start={self.start!r}, end={self.end!r})"

    def __sql_literal_insert__(self) -> str:
        return f"[{self.start}, {self.end}]"

    @classmethod
    def __sql_name__(self):
        return "RANGE1D"

    @classmethod
    def __sql_type__(self) -> sqltypes.DuckDBPyType:
        return duckdb.array_type(sqltypes.DOUBLE, 2)

    @classmethod
    def __create_type__(self) -> str:
        return "CREATE TYPE RANGE1D AS DOUBLE[2];"


class Range2D:
    def __init__(self, x_range: Range1D, y_range: Range1D):
        self.x_range = x_range
        self.y_range = y_range

    def __repr__(self) -> str:
        return f"Range2D(x_range={self.x_range.__repr__()!r}, y_range={self.y_range.__repr__()!r})"

    def __sql_literal_insert__(self) -> str:
        return f"{{'x_range': {self.x_range.__sql_literal_insert__()}, 'y_range': {self.y_range.__sql_literal_insert__()}}}"

    @classmethod
    def __sql_name__(self):
        return "RANGE2D"

    @classmethod
    def __sql_type__(self) -> sqltypes.DuckDBPyType:
        return duckdb.struct_type(
            {"x_range": Range1D.__sql_type__(), "y_range": Range1D.__sql_type__()}
        )

    @classmethod
    def __create_type__(self) -> str:
        return f"CREATE TYPE RANGE2D AS STRUCT(x_range {Range1D.__sql_type__()}, y_range {Range1D.__sql_type__()});"
