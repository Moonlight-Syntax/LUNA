import evaluate
from sacrebleu import BLEU

from luna.base import Metrics

# ---- n-gram based metrics ----


class BLEUMetrics(Metrics):
    def __init__(self) -> None:
        self.bleu_metrics = BLEU()

    def evaluate_example(self, hyp: str, ref: str) -> float:
        bleu_result = self.bleu_metrics.sentence_score(hyp, [ref])
        return bleu_result.score

    def __repr__(self) -> str:
        return "BLEU"


class METEORMetrics(Metrics):
    SCALE_FACTOR = 100.0

    def __init__(self) -> None:
        self.meteor_metrics = evaluate.load("meteor")

    def evaluate_example(self, hyp: str, ref: str) -> float:
        meteor_result = self.meteor_metrics.compute(predictions=[hyp], references=[ref])
        return meteor_result["meteor"] * self.SCALE_FACTOR

    def __repr__(self) -> str:
        return "METEOR"


class ROUGEMetrics(Metrics):
    """
    Attributes
    ----------
    setting: str
        Available options: ["1", "2", "L", "Lsum"].
    """

    SCALE_FACTOR = 100.0

    def __init__(self, setting: str = "1") -> None:
        self.setting = setting
        self.rouge_metrics = evaluate.load("rouge")

    def evaluate_example(self, hyp: str, ref: str) -> float:
        rouge_result = self.rouge_metrics.compute(predictions=[hyp], references=[ref])
        return rouge_result[f"rouge{self.setting}"] * self.SCALE_FACTOR

    def __repr__(self) -> str:
        return "ROUGE"


class CHRFMetrics(Metrics):
    def __init__(self, char_order: int = 6, word_order: int = 2, beta: int = 2) -> None:
        self.chrf_metrics = evaluate.load("chrf")
        self.chrf_args = {
            "char_order": char_order,
            "word_order": word_order,
            "beta": beta,
        }

    def evaluate_example(self, hyp: str, ref: str) -> float:
        score_dict = self.chrf_metrics.compute(
            predictions=[hyp],
            references=[[ref]],
            **self.chrf_args,
        )
        return score_dict["score"]

    def __repr__(self) -> str:
        return "CHRF"
