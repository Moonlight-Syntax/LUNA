import typing as tp

import pytest

from luna.ngram import BLEUMetrics, CHRFMetrics, METEORMetrics, ROUGEMetrics
from tests.conftest import get_mock_data


class TestNGramMetrics:
    EPS = 1e-2

    @pytest.mark.parametrize(
        ["metrics_class", "mean_answer"],
        (
            (BLEUMetrics, 35.20),
            (CHRFMetrics, 42.76),
            (METEORMetrics, 46.80),
        ),
    )
    def test_ngram_metrics(self, metrics_class: tp.Any, mean_answer: float) -> None:
        hyps, refs = get_mock_data()
        metrics = metrics_class()
        result = metrics.evaluate_batch(hyps, refs)
        mean_result = sum(result) / len(result)

        assert len(result) == len(hyps)
        assert (-self.EPS < min(result)) and (max(result) <= 100.0 + self.EPS)
        assert abs(mean_result - mean_answer) < self.EPS

    @pytest.mark.parametrize(
        ["setting", "mean_answer"],
        (
            ("1", 49.66),
            ("2", 39.17),
            ("L", 49.66),
        ),
    )
    def test_rouge(self, setting: str, mean_answer: float) -> None:
        hyps, refs = get_mock_data()
        metrics = ROUGEMetrics(setting=setting)
        result = metrics.evaluate_batch(hyps, refs)
        mean_result = sum(result) / len(result)

        assert len(result) == len(hyps)
        assert (-self.EPS < min(result)) and (max(result) <= 100.0 + self.EPS)
        assert abs(mean_result - mean_answer) < self.EPS
