import dataclasses
import typing as tp

import pytest

from luna.s3 import S3Metrics


class TestS3:
    @dataclasses.dataclass
    class TestCase:
        mode: str
        hyp: str
        ref: str
        expected_result: tp.Any

    @pytest.mark.parametrize("test_case", [
        TestCase(
            mode="pyr",
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            expected_result=pytest.approx(0.00271, abs=1e-5)
        ),
        TestCase(
            mode="resp",
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            expected_result=pytest.approx(0.13322, abs=1e-5)
        )
    ])
    def test_s3_evaluate_example(self, test_case: TestCase) -> None:
        result = S3Metrics(mode=test_case.mode).evaluate_example(test_case.hyp, test_case.ref)
        assert result == test_case.expected_result

    @dataclasses.dataclass
    class TestCaseBatch:
        mode: str
        hyps: tp.List[str]
        refs: tp.List[str]
        expected_results: tp.List[tp.Any]

    @pytest.mark.parametrize("test_case", [
        TestCaseBatch(
            mode="pyr",
            hyps=["Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
                  "Children should not be allowed to cry, even for the sake of the happiness of all mankind",
                  "Being able to forget is the basis for good mood and well-being"],
            refs=["What a good day today! Whether to go have a cup of tea or hang oneself.",
                  "The happiness of the world is not worth one tear on the cheek of an innocent child",
                  "Man is happy in his ability to forget"],
            expected_results=[pytest.approx(res, abs=1e-5) for res in [0.00271, -0.07147, -0.16247]]
        ),
        TestCaseBatch(
            mode="resp",
            hyps=["Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
                  "Children should not be allowed to cry, even for the sake of the happiness of all mankind",
                  "Being able to forget is the basis for good mood and well-being"],
            refs=["What a good day today! Whether to go have a cup of tea or hang oneself.",
                  "The happiness of the world is not worth one tear on the cheek of an innocent child",
                  "Man is happy in his ability to forget"],
            expected_results=[pytest.approx(res, abs=1e-5) for res in [0.13322, 0.09046, 0.07848]]
        ),
    ])
    def test_s3_evaluate_batch(self, test_case: TestCaseBatch) -> None:
        results = S3Metrics(mode=test_case.mode).evaluate_batch(test_case.hyps, test_case.refs)
        assert results == test_case.expected_results
