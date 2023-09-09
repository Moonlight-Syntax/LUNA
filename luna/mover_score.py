# pylint: disable=C0415,C0103
import typing as tp
from collections import defaultdict

from luna import utils
from luna.base import Metrics
from luna.sources.moverscore_utils import (
    get_idf_dict,
    get_model_and_tokenizer_from_transformers,
    word_mover_score,
)


class MoverScoreMetrics(Metrics):
    """
    Implementation of Mover Score metric, introduced in the paper
    'MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance', Zhao W. et al.

    Parameters
    ----------
    n_gram: int, default = 1
        n_gram size to use in mover score calculation; see Section 3.1 of paper for details.
        To calculate Sentence Mover Score (Mover Score with n_gram equal to the whole sentence)
        provide n_gram parameter greater than length of any sentence you will evaluate.
    model_name: str, default = "distilbert-base-uncased"
        Name of the model to get embeddings from.
        Can be any bert-like model from huggingface. To reproduce results from the paper use "textattack/bert-base-uncased-MNLI"
    compute_idfs: bool, default = False
        Whether to use idfs or give all words the same weight; see Section 3.1 of paper for details.
        Has effect only on evaluate_batch function.
    stop_words_file: str | None, default = None
        Path to file with space-separated list of stopwords. If set to None, no stopwords are used.
    remove_subwords: bool, default = True
        Whether to remove subword tokens before calculating n-grams.
    batch_size: int, default = 256
        Batch size for mover score calculation.
    device: str | torch.device, default = "cpu"
        Device for mover score calculation.

    Notes
    -----
    Code mostly taken from https://github.com/AIPHES/emnlp19-moverscore

    TODO: add multireference metric functionality.
    """

    def __init__(
        self,
        n_gram: int = 1,
        model_name: str = "distilbert-base-uncased",
        compute_idfs: bool = False,
        stop_words_file: tp.Optional[str] = None,
        remove_subwords: bool = True,
        batch_size: int = 256,
        device: tp.Union[str, "torch.device"] = "cpu",
    ) -> None:
        self.word_mover_score = word_mover_score

        if compute_idfs:
            self.get_idf_dict = get_idf_dict

        self.stop_words = []
        if stop_words_file is not None:
            with open(stop_words_file, "r", encoding="utf-8") as file:
                self.stop_words = file.read().strip().split()

        model, tokenizer = get_model_and_tokenizer_from_transformers(model_name, device)

        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.compute_idfs = compute_idfs
        self.n_gram = n_gram
        self.remove_subwords = remove_subwords
        self.batch_size = batch_size
        self.device = device

    def evaluate_example(self, hyp: str, ref: str) -> float:
        idf_dict_ref = defaultdict(lambda: 1.0)
        idf_dict_hyp = defaultdict(lambda: 1.0)

        score = self.word_mover_score(
            [ref],
            [hyp],
            idf_dict_ref,
            idf_dict_hyp,
            model_name=self.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            stop_words=self.stop_words,
            n_gram=self.n_gram,
            remove_subwords=self.remove_subwords,
            batch_size=1,
            device=self.device,
        )
        return score[0]

    def evaluate_batch(self, hyps: tp.List[str], refs: tp.Optional[tp.List[str]]) -> tp.List[float]:
        utils.validate_batch(hyps, refs)

        if self.compute_idfs:
            idf_dict_ref = self.get_idf_dict(refs, self.tokenizer)
            idf_dict_hyp = self.get_idf_dict(hyps, self.tokenizer)
        else:
            idf_dict_ref = defaultdict(lambda: 1.0)
            idf_dict_hyp = defaultdict(lambda: 1.0)

        return self.word_mover_score(
            refs,
            hyps,
            idf_dict_ref,
            idf_dict_hyp,
            model_name=self.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            stop_words=self.stop_words,
            n_gram=self.n_gram,
            remove_subwords=self.remove_subwords,
            batch_size=self.batch_size,
            device=self.device,
        )

    def __repr__(self) -> str:
        return "MoverScore"
