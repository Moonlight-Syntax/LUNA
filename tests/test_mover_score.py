import dataclasses
import logging
import typing as tp

import pytest
import torch

from luna.mover_score import MoverScoreMetrics

ABS = 1e-1


class TestMoverScore:
    @dataclasses.dataclass
    class CaseExample:
        metric_parameters: tp.Dict[str, tp.Any]
        hyp: str
        ref: str
        expected_result: tp.Any

    @dataclasses.dataclass
    class CaseBatch:
        metric_parameters: tp.Dict[str, tp.Any]
        hyps: tp.List[str]
        refs: tp.List[str]
        expected_result: tp.Any

    @pytest.mark.parametrize("test_case", [
        CaseExample(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            metric_parameters=dict(
                n_gram=1,
                model_name="distilbert-base-uncased",
                compute_idfs=False,
                stop_words_file=None,
                remove_subwords=True,
                batch_size=256,
                device="cpu"
            ),
            expected_result=pytest.approx(0.47848, abs=ABS)
        ),
        CaseExample(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            metric_parameters=dict(
                n_gram=1,
                model_name="textattack/bert-base-uncased-MNLI",
                compute_idfs=False,
                stop_words_file=None,
                remove_subwords=True,
                batch_size=256,
                device="cpu"
            ),
            expected_result=pytest.approx(0.31345, abs=ABS)
        ),
        CaseExample(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            metric_parameters=dict(
                n_gram=2,
                model_name="distilbert-base-uncased",
                compute_idfs=False,
                stop_words_file=None,
                remove_subwords=True,
                batch_size=256,
                device="cpu"
            ),
            expected_result=pytest.approx(0.58964, abs=ABS)
        ),
        CaseExample(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            metric_parameters=dict(
                n_gram=1e9,  # on the whole sentence
                model_name="distilbert-base-uncased",
                compute_idfs=False,
                stop_words_file=None,
                remove_subwords=True,
                batch_size=256,
                device="cpu"
            ),
            expected_result=pytest.approx(0.90059, abs=ABS)
        ),
        CaseExample(
            hyp="Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
            ref="What a good day today! Whether to go have a cup of tea or hang oneself.",
            metric_parameters=dict(
                n_gram=1,
                model_name="distilbert-base-uncased",
                compute_idfs=False,
                stop_words_file=None,
                remove_subwords=True,
                batch_size=256,
                device="cuda"
            ),
            expected_result=pytest.approx(0.47848, abs=ABS)
        ),
    ])
    def test_mover_score_evaluate_example(self, test_case: CaseExample) -> None:
        if test_case.metric_parameters["device"] == "cuda" and not torch.cuda.is_available():
            logging.warning("test did not run because cuda is not available")
            return
        metric = MoverScoreMetrics(**test_case.metric_parameters)
        result = metric.evaluate_example(test_case.hyp, test_case.ref)
        assert result == test_case.expected_result

    @pytest.mark.parametrize("test_case", [
        CaseBatch(
            hyps=["Today is such a great day! I'm not sure if I want to go get some tea or just end it all.",
                  "Children should not be allowed to cry, even for the sake of the happiness of all mankind",
                  "Being able to forget is the basis for good mood and well-being"],
            refs=["What a good day today! Whether to go have a cup of tea or hang oneself.",
                  "The happiness of the world is not worth one tear on the cheek of an innocent child",
                  "Man is happy in his ability to forget"],
            metric_parameters=dict(
                n_gram=1,
                model_name="distilbert-base-uncased",
                compute_idfs=True,
                stop_words_file=None,
                remove_subwords=True,
                batch_size=256,
                device="cpu"
            ),
            expected_result=[pytest.approx(i, abs=ABS) for i in [0.46791, 0.42041, 0.37455]]
        )
    ])
    def test_mover_score_evaluate_batch(self, test_case: CaseBatch) -> None:
        if test_case.metric_parameters["device"] == "cuda" and not torch.cuda.is_available():
            logging.warning("test did not run because cuda is not available")
            return
        metric = MoverScoreMetrics(**test_case.metric_parameters)
        result = metric.evaluate_batch(test_case.hyps, test_case.refs)
        assert result == test_case.expected_result
