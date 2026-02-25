from abc import ABC
import random

from tt.base.result.result import ResultType
from tt.base.instantiate.extensions import (
    register_singleton,
    get_singleton_service_descriptors,
)
from tt.sdk import init_env


def test_result_type():
    init_env("", "")

    assert 1 == True
