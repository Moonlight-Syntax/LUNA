import dataclasses
import typing as tp

import pytest

from luna.rouge_we import RougeWeMetrics


class TestRougeWe:
    @dataclasses.dataclass
    class TestCase:
        hyp: str
        ref: str
        expected_result: tp.Any

    TEST_CASES = [
        TestCase(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            expected_result=pytest.approx(0.34286, abs=1e-5)
        )
    ]

    @staticmethod
    @pytest.fixture
    def rouge_we_metric() -> RougeWeMetrics:
        metric = RougeWeMetrics(emb_path=None, n_gram=3, stem=True, tokenize=True)
        return metric

    @pytest.mark.parametrize("test_case", TEST_CASES)
    def test_bart_score_evaluate_example(self, test_case: TestCase, rouge_we_metric) -> None:
        result = rouge_we_metric.evaluate_example(test_case.hyp, test_case.ref)
        assert result == test_case.expected_result
