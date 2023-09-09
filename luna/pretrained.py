import logging
import typing as tp

import evaluate
import torch

from luna.base import Metrics
from luna.sources.bart_score import BARTScorer

# ---- Pre-trained neural-based metrics ----


class BERTScoreMetrics(Metrics):
    """
    BERTScore implementation.
    Based on the HF implementation: https://huggingface.co/spaces/evaluate-metric/bertscore

    Attributes
    ----------
    model_name: str
        The model which is pulled from the HuggingFace model hub.
        Alternative: path to a locally pre-saved model.
    bert_score_args: dict from str to Any
        Arguments for the BERTScore metric.
        Key-value pairs:
        * baseline_path: str
          Path to the baseline file.
        * rescale_with_baseline: bool
          It's highly reccomended to use the baseline rescaling option.
        * use_fast_tokenizer: bool, default is False
          Fast tokenization option.
        * idf: bool, default is False
          A flag to indicate whether to use idf weighting or not.
        * lang: str
          Natural language behind the metric's computation.
        * device: str
          Device for the inference process.
    bert_score_metrics: evaluate.Metric
        The BERTScore metric implementation from HuggingFace / evaluate.
    loaded: bool, default is False
        A flag to indicate whether the metric is loaded.
        Use Metrics.load() before Metrics.evaluate_example() or Metrics.evaluate_batch().
    mode: str, default is "f1"
        Mode of evaluation. Can be "f1", "precision" or "recall"
    """

    def __init__(
        self,
        model_name: str = "bertscore",
        baseline_path: str = None,
        rescale_with_baseline: bool = False,
        use_fast_tokenizer: bool = False,
        idf: bool = False,
        lang: str = "en",
        device: str = "cpu",
        mode: str = "f1",
    ) -> None:
        if not baseline_path and rescale_with_baseline:
            raise ValueError("For baseline rescaling provide the path to the baseline implementation")
        if mode not in ["f1", "precision", "recall"]:
            raise ValueError(f"mode argument should be one of: f1, precision, recall. Got mode={mode}")
        self.bert_score_metrics = None
        self.model_name = model_name
        self.bert_score_args = {
            "baseline_path": baseline_path,
            "rescale_with_baseline": rescale_with_baseline,
            "use_fast_tokenizer": use_fast_tokenizer,
            "idf": idf,
            "lang": lang,
            "device": device,
        }
        self.loaded = False
        self.mode = mode
        if mode not in ["f1", "precision", "recall"]:
            raise ValueError("An attribute mode should satisfy to a restricted list of values")

    def evaluate_example(self, hyp: str, ref: str) -> float:
        if not self.loaded:
            raise ValueError("The metric is not loaded, use Metrics.load() before Metrics.evaluate_example()")

        scores = self.bert_score_metrics.compute(predictions=[hyp], references=[ref], **self.bert_score_args)
        logging.log(logging.INFO, f"hashcode: {scores['hashcode']}")
        return scores[self.mode][0]

    def evaluate_batch(self, hyps: tp.List[str], refs: tp.List[str]) -> tp.List[float]:
        if not self.loaded:
            raise ValueError("The metric is not loaded, use Metrics.load() before Metrics.evaluate_batch()")

        scores = self.bert_score_metrics.compute(predictions=hyps, references=refs, **self.bert_score_args)
        logging.log(logging.INFO, f"hashcode: {scores['hashcode']}")
        return scores.get(self.mode, [0] * len(hyps))

    def load(self) -> None:
        self.bert_score_metrics = evaluate.load(self.model_name)
        self.loaded = True

    def __repr__(self) -> str:
        return "BERTScore"


class BARTScoreMetrics(Metrics):
    """
    Implementation of the metric introduced in the paper
    'BARTScore: Evaluating Generated Text as Text Generation', Yuan W. et al.

    Parameters
    ----------
    checkpoint: str, default="facebook/bart-large-cnn"
        Name of checkpoint from pretrained models from huggingface transformers.
        Will use this model to calculate score.
    tokenizer_checkpoint: str | None, default=None
        Name of pretrained checkpoint to load from huggingface.
        If set to None, will default to model checkpoint.
    max_length: int, default=1024
        Truncate the sentences longer than max_length.
    device: str | torch.device, default="cpu"
        Name of device on which to perform evaluation.
    """

    def __init__(
        self,
        checkpoint: str = "facebook/bart-large-cnn",
        tokenizer_checkpoint: tp.Optional[str] = None,
        max_length: int = 1024,
        device: tp.Union[str, torch.device] = "cpu",
    ) -> None:
        self.bart_scorer = BARTScorer(
            checkpoint=checkpoint, tokenizer_checkpoint=tokenizer_checkpoint, max_length=max_length, device=device
        )

    def evaluate_example(self, hyp: str, ref: str) -> float:
        scores = self.bart_scorer.score(hyp, ref, batch_size=1)
        return scores[0]

    def evaluate_batch(self, hyps: tp.List[str], refs: tp.Optional[tp.List[str]]) -> tp.List[float]:
        assert refs is not None, "This is a reference-based metric, please provide references"
        assert len(hyps) == len(
            refs
        ), f"Length of hypotheses sequence ({len(hyps)}) must be equal to length of references sequence ({len(refs)})"
        return self.bart_scorer.score(hyps, refs, batch_size=len(hyps))

    def __repr__(self) -> str:
        return "BARTScore"
