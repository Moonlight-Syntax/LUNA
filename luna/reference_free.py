import typing as tp

from luna.base import Metrics
from luna.sources.data_stats_metrics import DataStatsMetric


# ---- Reference-free statistics ----
class ReferenceFreeMetrics(Metrics):
    """
    Calculates extractive statistics such as coverage, density, compression as
    defined in Newsroom paper as well as the percentage of novel n-grams in the
    summary vs the input text and the percentage of n-grams in the summary which are repeated
    Important note: these statistics are meant to be calculated with respect to the source text (e.g. news article) as opposed to the reference.

    Attributes
    ----------
    n_gram: int, default is 3
        N-gram number.
    n_workers: int, default is 24
        Number of workers for the distributed computation.
    case: bool, default is False
        Flag indicating case-sensitiveness.
    tokenize: bool, default is True
        Flag indicating the tokenization.

    Methods
    -------
    evaluate_example(input_text: str, hyp: str)
        This method is provided for the sample evaluation.
    evaluate_batch(input_texts: tp.List[str], hyps: tp.List[str])
        This method is provided for the batch computation.
    """

    def __init__(self, n_gram: int = 3, n_workers: int = 24, case: bool = False, tokenize: bool = True) -> None:
        self.data_stats = DataStatsMetric(n_gram=n_gram, n_workers=n_workers, case=case, tokenize=tokenize)

        # TODO: changes of the following attributes should trigger the re-initialization of self.data_stats instance
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.case = case
        self.tokenize = tokenize

    @property
    def stats_key(self) -> str:
        raise NotImplementedError

    def evaluate_example(self, input_text: str, hyp: str) -> float:
        computed_dict = self.data_stats.evaluate_example(input_text=input_text, summary=hyp)
        return computed_dict[self.stats_key]

    def evaluate_batch(self, input_texts: tp.List[str], hyps: tp.List[str]) -> tp.List[float]:
        computed_batch = self.data_stats.evaluate_batch(input_texts=input_texts, summaries=hyps, aggregate=False)
        metrics_list = [metrics_dict[self.stats_key] for metrics_dict in computed_batch]
        return metrics_list

    def __repr__(self) -> str:
        raise NotImplementedError


class CompressionMetrics(ReferenceFreeMetrics):
    """
    Calculates compression ratio as one of the possible extractive statistics.
    """

    @property
    def stats_key(self) -> str:
        return "compression"

    def __repr__(self) -> str:
        return "Compression"


class CoverageMetrics(ReferenceFreeMetrics):
    """
    Calculates coverage ratio as one of the possible extractive statistics.
    """

    @property
    def stats_key(self) -> str:
        return "coverage"

    def __repr__(self) -> str:
        return "Coverage"


class LengthMetrics(ReferenceFreeMetrics):
    """
    Calculates compression ratio as one of the possible extractive statistics.
    """

    @property
    def stats_key(self) -> str:
        return "summary_length"

    def __repr__(self) -> str:
        return "Length"


class NoveltyMetrics(ReferenceFreeMetrics):
    """
    Calculates novelty ratio as one of the possible extractive statistics.
    """

    @property
    def stats_key(self) -> str:
        return f"percentage_novel_{self.n_gram}-gram"

    def __repr__(self) -> str:
        return "Novelty"


class DensityMetrics(ReferenceFreeMetrics):
    """
    Calculates density ratio as one of the possible extractive statistics.
    """

    @property
    def stats_key(self) -> str:
        return "density"

    def __repr__(self) -> str:
        return "Density"


class RepetitionMetrics(ReferenceFreeMetrics):
    """
    Calculates repetition ratio as one of the possible extractive statistics.
    """

    @property
    def stats_key(self) -> str:
        return f"percentage_repeated_{self.n_gram}-gram_in_summ"

    def __repr__(self) -> str:
        return "Repetition"
