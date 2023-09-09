import time
import typing as tp

import pytest

from luna.summaqa import SummaQAMetrics
from tests.conftest import get_mock_data

HYPS, REFS = get_mock_data(lang="ru")
ABS = 1e-3


class TestSummaQA:
    @pytest.mark.parametrize(
        ["indices", "gt_value"],
        (
            ([0, 2, 3, 5], 0.25241),
            ([1, 4], 0.4245),
        ),
    )
    def test_evaluate_batch(self, indices: tp.List[int], gt_value: float) -> None:
        metrics = SummaQAMetrics(lang="ru")

        hyps = [HYPS[ind] for ind in indices]
        refs = [REFS[ind] for ind in indices]
        metrics = metrics.evaluate_corpus(hyps=hyps, refs=refs)
        assert metrics == pytest.approx(gt_value, abs=ABS)

    @pytest.mark.performance
    def test_performance(self) -> None:
        metrics = SummaQAMetrics(lang="ru")

        MULTIPLIER = 30
        HYPS_LARGE = HYPS * MULTIPLIER
        REFS_LARGE = REFS * MULTIPLIER

        start = time.time()
        metrics = metrics.evaluate_corpus(hyps=HYPS_LARGE, refs=REFS_LARGE)
        end = time.time()

        mean_time = (end - start) / len(HYPS)
        assert mean_time < 2.0
