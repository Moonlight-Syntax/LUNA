import typing as tp

import pytest

from luna.calculate import Calculator
from luna.reference_free import LengthMetrics
from luna.statistical_measures import DepthScoreMetrics
from tests.conftest import get_mock_data


class TestCalculator:
    HYPS, REFS = get_mock_data()
    DEPTH_SCORE_METRICS = DepthScoreMetrics()
    LENGTH_METRICS = LengthMetrics(n_workers=1)

    @pytest.mark.parametrize(
        ["indices", "metrics_name", "gt_value"],
        (
            ([0, 1], "DepthScore", 0.08653),
            ([2, 3], "Length", 6),
        ),
    )
    def test_calculate_sequentially(self, indices: tp.List[int], metrics_name: str, gt_value: tp.Union[float, int]) -> None:
        calculator = Calculator(execute_parallel=False)

        hyps = [self.HYPS[ind] for ind in indices]
        refs = [self.REFS[ind] for ind in indices]

        metrics_dict = calculator.calculate(
            metrics=[self.DEPTH_SCORE_METRICS, self.LENGTH_METRICS],
            hyps=hyps,
            refs=refs
        )
        assert metrics_name in metrics_dict
        assert abs(min(metrics_dict[metrics_name]) - gt_value) < 1e-3

    @pytest.mark.parametrize(
        ["indices", "metrics_name", "gt_value"],
        (
            ([0, 1], "DepthScore", 0.0865),
            ([2, 3], "Length", 6),
        ),
    )
    def test_calculate_parallel(self, indices: tp.List[int], metrics_name: str, gt_value: tp.Union[float, int]) -> None:
        calculator = Calculator(execute_parallel=True)

        hyps = [self.HYPS[ind] for ind in indices]
        refs = [self.REFS[ind] for ind in indices]

        metrics_dict = calculator.calculate(
            metrics=[self.DEPTH_SCORE_METRICS, self.LENGTH_METRICS],
            hyps=hyps,
            refs=refs
        )
        assert metrics_name in metrics_dict
        assert abs(min(metrics_dict[metrics_name]) - gt_value) < 1e-3
