import dataclasses
import typing as tp

import pytest

from luna.blanc import BlancMetrics


class TestBlanc:
    @dataclasses.dataclass
    class CaseExample:
        hyp: str
        inp: str
        expected_result: tp.Any

    @pytest.mark.parametrize("test_case", [
        CaseExample(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            inp="What a good day today! Whether to go have a cup of tea or hang oneself.",
            expected_result=pytest.approx(-0.28571, abs=1e-5)
        ),
        CaseExample(
            hyp="Children should not be allowed to cry, even for the sake of the happiness of all mankind",
            inp="The happiness of the world is not worth one tear on the cheek of an innocent child",
            expected_result=pytest.approx(-0.28571, abs=1e-5)
        ),
        CaseExample(
            hyp="Being able to forget is the basis for good mood and well-being",
            inp="Man is happy in his ability to forget",
            expected_result=pytest.approx(0.33333, abs=1e-5)
        )
    ])
    def test_blanc_tune_evaluate_example(self, test_case: CaseExample) -> None:
        result = BlancMetrics(type="tune", device="cpu", random_seed=42).evaluate_example(test_case.hyp, test_case.inp)
        assert result == test_case.expected_result

    @pytest.mark.parametrize("test_case", [
        CaseExample(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            inp="What a good day today! Whether to go have a cup of tea or hang oneself.",
            expected_result=pytest.approx(0.0, abs=1e-5)
        ),
        CaseExample(
            hyp="Children should not be allowed to cry, even for the sake of the happiness of all mankind",
            inp="The happiness of the world is not worth one tear on the cheek of an innocent child",
            expected_result=pytest.approx(0.0, abs=1e-5)
        ),
        CaseExample(
            hyp="Being able to forget is the basis for good mood and well-being",
            inp="Man is happy in his ability to forget",
            expected_result=pytest.approx(0.0, abs=1e-5)
        )
    ])
    def test_blanc_help_evaluate_example(self, test_case: CaseExample) -> None:
        result = BlancMetrics(type="help", device="cpu").evaluate_example(test_case.hyp, test_case.inp)
        assert result == test_case.expected_result

    @dataclasses.dataclass
    class CaseBatch:
        hyps: tp.List[str]
        inps: tp.List[str]
        expected_results: tp.Any

    @pytest.mark.parametrize("test_case", [
        CaseBatch(
            hyps=["Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
                  "Children should not be allowed to cry, even for the sake of the happiness of all mankind",
                  "Being able to forget is the basis for good mood and well-being"],
            inps=["What a good day today! Whether to go have a cup of tea or hang oneself.",
                  "The happiness of the world is not worth one tear on the cheek of an innocent child",
                  "Man is happy in his ability to forget"],
            expected_results=[pytest.approx(i, abs=1e-5) for i in [0.0, 0.0, 0.0]]
        )
    ])
    def test_blanc_help_evaluate_batch(self, test_case: CaseBatch) -> None:
        results = BlancMetrics(type="help", device="cpu").evaluate_batch(test_case.hyps, test_case.inps)
        assert results == test_case.expected_results

    @pytest.mark.parametrize("test_case", [
        CaseBatch(
            hyps=["Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
                  "Children should not be allowed to cry, even for the sake of the happiness of all mankind",
                  "Being able to forget is the basis for good mood and well-being"],
            inps=["What a good day today! Whether to go have a cup of tea or hang oneself.",
                  "The happiness of the world is not worth one tear on the cheek of an innocent child",
                  "Man is happy in his ability to forget"],
            expected_results=[pytest.approx(i, abs=1e-5) for i in [-0.28571, -0.28571, 0.33333]]
        )
    ])
    def test_blanc_tune_evaluate_batch(self, test_case: CaseBatch) -> None:
        results = BlancMetrics(type="tune", device="cpu", random_seed=42).evaluate_batch(test_case.hyps, test_case.inps)
        assert results == test_case.expected_results
