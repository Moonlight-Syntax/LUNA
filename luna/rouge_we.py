# pylint: disable=C0103
import typing as tp

from luna.base import Metrics
from luna.sources.rouge_we_utils import load_embeddings, rouge_n_we


class RougeWeMetrics(Metrics):
    """
    Implementation of the metric introduced in the paper
    'Better summarization evaluation with word embeddings for ROUGE', Ng J. P., Abrecht V.

    Parameters
    ----------
    emb_path: str, default = None
        Path to file with word embeddings. If set to None, will download files from
        https://drive.google.com/uc?id=1NGAoXi_QzpXl-gAon2UPwpX_PnxYupjn which is a copy of the file from
        http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2
        The file will be downloaded to the local directory "embeddings/" (created if not exists),
        and will have a name "deps.words"
    n_gram: int, default=3
        N_gram length to be used for calculation;
        If n_gram=3, only calculates ROUGE-WE for n=3;
        Reset n_gram to calculate for other n-gram lengths
    alpha: float, default=0.5
        Controls the weight of recall and precision in the calculated f-beta-score.
        The bigger alpha is, the more important is precision compared to recall.
        alpha should be between 0. and 1. inclusively.
        alpha = 1 / (1 + beta^2) where beta is a parameter from f-beta score.
        When alpha=0.5, the resulting f-score is a normal f1-score.
    stem: bool, default=True
        Whether to apply stemming to input;
        Otherwise assumes that user has done any necessary tokenization.
    tokenize: bool, default=True
        Whether to apply basic tokenization to input;
        Otherwise assumes that user has done any necessary tokenization.

    Notes
    -----
    The majority of code was taken from https://github.com/Yale-LILY/SummEval
    and a little bit from https://github.com/UKPLab/emnlp-ws-2017-s3/tree/b524407ada525c81ceacd2590076e20103213e3b

    TODO: add multireference metric functionality.
    """

    def __init__(
        self,
        emb_path: tp.Optional[str] = None,
        n_gram: int = 3,
        alpha: float = 0.5,
        stem: bool = True,
        tokenize: bool = True,
    ):
        self.word_embeddings = load_embeddings(emb_path)
        self.n_gram = n_gram
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be a float number between 0. and 1.")
        self.alpha = alpha
        self.tokenize = tokenize
        self.stem = stem

    def evaluate_example(self, hyp: str, ref: str) -> float:
        score = rouge_n_we(
            [hyp],
            [ref],
            self.word_embeddings,
            self.n_gram,
            alpha=self.alpha,
            return_all=True,
            stem=self.stem,
            tokenize=self.tokenize,
        )
        score_dict = {
            f"rouge_we_{self.n_gram}_p": score[0],
            f"rouge_we_{self.n_gram}_r": score[1],
            f"rouge_we_{self.n_gram}_f": score[2],
        }
        return score_dict[f"rouge_we_{self.n_gram}_f"]

    def __repr__(self) -> str:
        return "RougeWe"
