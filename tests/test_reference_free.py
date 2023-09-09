import typing as tp

import pytest

from luna.reference_free import (
    CompressionMetrics, CoverageMetrics, LengthMetrics, NoveltyMetrics, RepetitionMetrics
)
from tests.conftest import get_mock_data


class TestReferenceFreeMetrics:
    HYPS, INPUT_TEXTS = get_mock_data()

    @pytest.mark.parametrize(
        ["metrics_class", "gt_value"],
        (
            (CompressionMetrics, 0.8571),
            (CoverageMetrics, 0.5714),
            (LengthMetrics, 7.0),
            (NoveltyMetrics, 0.6),
            (RepetitionMetrics, 0.00),
        ),
    )
    def test_evaluate_example(self, metrics_class: tp.Any, gt_value: float) -> None:
        metrics = metrics_class()
        metrics_value = metrics.evaluate_example(input_text=self.INPUT_TEXTS[0], hyp=self.HYPS[0])
        assert metrics_value == pytest.approx(gt_value, abs=1e-3)

    @pytest.mark.parametrize(
        ["metrics_class", "min_value", "max_value"],
        (
            (CompressionMetrics, 0.8333, 1.3333),
            (CoverageMetrics, 0.25, 1.0),
            (LengthMetrics, 4.0, 9.0),
            (NoveltyMetrics, 0.0, 1.0),
            (RepetitionMetrics, 0.00, 0.00),
        ),
    )
    def test_evaluate_batch(self, metrics_class: tp.Any, min_value: float, max_value: float) -> None:
        metrics = metrics_class()
        metrics_list = metrics.evaluate_batch(input_texts=self.INPUT_TEXTS, hyps=self.HYPS)

        assert len(metrics_list) == len(self.HYPS)
        assert min(metrics_list) == pytest.approx(min_value, abs=1e-3) and \
            max(metrics_list) == pytest.approx(max_value, abs=1e-3)
