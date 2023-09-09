import dataclasses
import typing as tp

import pytest

from luna.base import Metrics
from tests.conftest import get_mock_data


class ReferenceBasedMetric(Metrics):
    def evaluate_example(self, hyp: str, ref: tp.Optional[str]) -> float:
        return len(hyp) - len(ref)


class ReferenceFreeMetric(Metrics):
    def evaluate_example(self, hyp: str, ref: tp.Optional[str]) -> float:
        return len(hyp) - 10.0


@dataclasses.dataclass
class Case:
    metric: Metrics
    hyps: tp.List[str]
    refs: tp.Optional[tp.List[str]]
    expected_result: tp.List[float]


BASIC_TEST_CASES = [
    Case(metric=ReferenceBasedMetric(),
         hyps=['cc', 'bb'],
         refs=['aa', 'bbb'],
         expected_result=[0., -1.]),
    Case(metric=ReferenceFreeMetric(),
         hyps=['cc', 'bbb'],
         refs=None,
         expected_result=[-8., -7.])
]


@pytest.mark.parametrize('test_case', BASIC_TEST_CASES)
def test_base_class_compute(test_case: Case) -> None:
    result = test_case.metric.evaluate_batch(test_case.hyps, test_case.refs)
    assert result == test_case.expected_result


HYPS, REFS = get_mock_data()
BATCH_TEST_CASES = [
    Case(metric=ReferenceBasedMetric(),
         hyps=HYPS,
         refs=REFS,
         expected_result=[6]),
    Case(metric=ReferenceFreeMetric(),
         hyps=HYPS,
         refs=REFS,
         expected_result=[6])
]


@pytest.mark.parametrize('test_case', BATCH_TEST_CASES)
def test_mock_data_base_compute(test_case: Case) -> None:
    result = test_case.metric.evaluate_batch(test_case.hyps, test_case.refs)
    # Comparison of lengths
    assert len(result) == test_case.expected_result[0]
