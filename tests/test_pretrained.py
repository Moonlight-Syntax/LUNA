import dataclasses
import typing as tp

import pytest

from luna.pretrained import BARTScoreMetrics, BERTScoreMetrics
from tests.conftest import get_mock_data


class TestBARTScore:
    @dataclasses.dataclass
    class BARTScoreCase:
        hyp: str
        ref: str
        expected_result: tp.Any

    BARTSCORE_CASES = [
        BARTScoreCase(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            expected_result=pytest.approx(-6.6539, abs=1e-4)
        )
    ]

    @staticmethod
    @pytest.fixture
    def bart_score_metric() -> BARTScoreMetrics:
        # Create a smaller model, not large
        metric = BARTScoreMetrics(checkpoint="ainize/bart-base-cnn",
                                  tokenizer_checkpoint="facebook/bart-base")
        return metric

    @pytest.mark.parametrize("test_case", BARTSCORE_CASES)
    def test_bart_score_evaluate_example(self, test_case: BARTScoreCase, bart_score_metric) -> None:
        result = bart_score_metric.evaluate_example(test_case.hyp, test_case.ref)
        assert result == test_case.expected_result


class TestBERTScore:
    EPS = 1e-2
    MODEL_NAME = "bertscore"
    DEFAULT_PARAMS = {
        "baseline_path": None,
        "rescale_with_baseline": False,
        "use_fast_tokenizer": False,
        "lang": "en",
        "device": "cpu",
    }

    @classmethod
    def modify_params(cls, dict_vals: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        params = cls.DEFAULT_PARAMS.copy()
        for key, value in dict_vals.items():
            params[key] = value
        return params

    @pytest.mark.parametrize(
        ["parameters_dict", "mean_answer"],
        (
                (
                        DEFAULT_PARAMS,
                        1.00
                ),
        ),
    )
    def test_bert_score(self, parameters_dict: tp.Dict[str, tp.Any], mean_answer: float) -> None:
        hyps, refs = get_mock_data()
        bert_score_metrics = BERTScoreMetrics(model_name=self.MODEL_NAME, **parameters_dict)
        bert_score_metrics.load()
        scores = bert_score_metrics.evaluate_batch(hyps, refs)

        assert len(scores) == len(hyps)
