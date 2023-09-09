import typing as tp
from abc import ABC, abstractmethod

from tqdm import tqdm

from luna import utils


class Metrics(ABC):
    """
    A base class to calculate metrics.
    """

    def __repr__(self) -> str:
        raise NotImplementedError

    def evaluate_batch(self, hyps: tp.List[str], refs: tp.Optional[tp.List[str]]) -> tp.List[float]:
        """
        Basic iteration over samples.
        Compute metrics for a dataset.

        Parameters
        ----------

        Returns
        -------
        list of float
            A list of metrics values.
        """
        # the following loop can be parallelized
        # however, as we assume the gpu-based inference, we can't wrap it into a simple joblib.Parallel etc.
        utils.validate_batch(hyps, refs)

        metrics_list = []
        for i, hyp in tqdm(enumerate(hyps)):
            ref = refs[i] if refs else None
            metrics_list.append(self.evaluate_example(hyp, ref))
        return metrics_list

    @abstractmethod
    def evaluate_example(self, hyp: str, ref: tp.Optional[str]) -> float:
        """
        Compute metrics for one sample.

        Parameters
        ----------
        hyp: str
            Hypothesis (generated sentence).
        ref: str, optional
            Reference (ground-truth sentence).

        Returns
        -------
        float
            Metrics calculated for the example.
        """
        raise NotImplementedError
