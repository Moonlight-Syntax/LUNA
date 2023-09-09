import typing as tp

import torch

from luna.base import Metrics
from luna.sources.bary_score import BaryScoreMetricsInternal
from luna.sources.depth_score import DepthScoreMetricsInternal
from luna.sources.info_lm import InfoLMMetricsInternal

# ---- SummEval paper metrics (Wasserstein-based) ----


class BaryScoreMetrics(Metrics):
    """
    Automatic Text Evaluation through the Lens of Wasserstein Barycenters.
    Reference: https://arxiv.org/abs/2108.12463

    Attributes
    ----------
    model_name: str, default is "bert-base-uncased"
        Pretrained transformer behind the metric.
        Check out: https://huggingface.co/bert-base-uncased
    bary_score_key: str, optional
        Bary score provides metrics computed for various parameters.
        By default we consider "baryscore_W".
    last_layers: int, optional
        N. of last layers to use in the pretrained model.
    use_idfs: bool
        If True, use idf costs. Otherwise, use uniform weights.
    sinkhorn_ref: float
        Weight of the KL in the SD.
    device: str, optional
        Device to be used for inference. By default, try to choose an available GPU.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        bary_score_key: str = "baryscore_W",
        last_layers: int = 5,
        use_idfs: bool = True,
        sinkhorn_ref: float = 0.01,
        device: tp.Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.bary_score_key = bary_score_key
        self.last_layers = last_layers
        self.use_idfs = use_idfs
        self.sinkhorn_ref = sinkhorn_ref
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: changes of the previous attributes should trigger the re-initialization of self.bary_score_metrics instance
        self.bary_score_metrics = BaryScoreMetricsInternal(
            model_name=model_name, last_layers=last_layers, use_idfs=use_idfs, sinkhorn_ref=sinkhorn_ref, device=device
        )

    def _run_bary_score(self, hyps: tp.List[str], refs: tp.List[str]) -> tp.Dict[str, tp.Any]:
        idf_dict_hyp, idf_dict_ref = self.bary_score_metrics.prepare_idfs(hyps, refs)
        metrics_dict = self.bary_score_metrics.evaluate_batch(hyps, refs, idf_dict_hyp, idf_dict_ref)
        return metrics_dict

    def evaluate_example(self, hyp: str, ref: str) -> float:
        metrics_dict = self._run_bary_score([hyp], [ref])
        return metrics_dict[self.bary_score_key][0]

    def evaluate_batch(self, hyps: tp.List[str], refs: tp.List[str]) -> tp.List[float]:
        metrics_dict = self._run_bary_score(hyps, refs)
        return metrics_dict[self.bary_score_key]

    def __repr__(self) -> str:
        return "BaryScore"


class DepthScoreMetrics(Metrics):
    """
    A Pseudo-Metric between Probability Distributions based on Depth-Trimmed Regions.
    Reference: https://arxiv.org/abs/2103.12711

    Attributes
    ----------
    model_name: str, default is "bert-base-uncased"
        Pretrained transformer behind the metric.
        Check out: https://huggingface.co/bert-base-uncased
    layers_to_consider: int
        N. of layers to use in the pretrained model.
    considered_measure: str
        Measure to choose.
        Possible values: ["irw", "ai_irw", "wasserstein", "sliced", "mmd"].
    device: str, optional
        Device to be used for inference. By default, try to choose an available GPU.

    TODO: perhaps we'll need to support (p, eps, n_alpha) parameters.
    """

    DEPTH_SCORE_KEY: str = "depth_score"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        layers_to_consider: int = 9,
        considered_measure: str = "irw",
        device: tp.Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.layers_to_consider = layers_to_consider
        self.considered_measure = considered_measure
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.depth_score_metrics = DepthScoreMetricsInternal(
            model_name=model_name,
            layers_to_consider=layers_to_consider,
            considered_measure=considered_measure,
            device=self.device,
        )

    def evaluate_example(self, hyp: str, ref: str) -> float:
        # NOTE: due to the lack of implementation in DepthScoreMetricsInternal
        # we have to call evaluate_batch to handle one example
        sample_batch = self.evaluate_batch(hyps=[hyp], refs=[ref])
        return sample_batch[0]

    def evaluate_batch(self, hyps: tp.List[str], refs: tp.List[str]) -> tp.List[float]:
        metrics_dict = self.depth_score_metrics.evaluate_batch(batch_hyps=hyps, batch_refs=refs)
        return metrics_dict[self.DEPTH_SCORE_KEY]

    def __repr__(self) -> str:
        return "DepthScore"


class InfoLMMetrics(Metrics):
    """
    InfoLM: A New Metric to Evaluate Summarization & Data2Text Generation.
    Reference: https://arxiv.org/abs/2112.01589

    Attributes
    ----------
    model_name: str, default is "bert-base-uncased"
        Pretrained transformer behind the metric.
        Check out: https://huggingface.co/bert-base-uncased
    temperature: float, default is 0.25
        Temperature to calibrate the LM.
    measure_to_use: str, default is "fisher_rao"
        Measure of information. Available options: ["kl", "alpha", "renyi", "beta", "ab", "l1", "l2", "linf", "fisher_rao"].
    use_idf_weights: bool, default is True
        If True, use idf costs. Otherwise, use uniform weights.
    alpha: float, optional
        Alpha parameter in the internal measure function.
        Note: not every function supports this parameter. Used in "ab", "alpha" or "renyi".
    beta: float, optional
        Beta parameter in the internal measure function.
        Note: not every function supports this parameter. Used in "ab", "beta".
    device: str, optional
        Device to be used for inference. By default, try to choose an available GPU.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        temperature: float = 0.25,
        measure_to_use: str = "fisher_rao",
        use_idf_weights: bool = True,
        alpha: tp.Optional[float] = None,
        beta: tp.Optional[float] = None,
        device: tp.Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.measure_to_use = measure_to_use
        self.use_idf_weights = use_idf_weights
        self.alpha = alpha
        self.beta = beta
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if measure_to_use not in ["kl", "alpha", "renyi", "beta", "ab", "l1", "l2", "linf", "fisher_rao"]:
            raise ValueError("An attribute measure_to_use should satisfy to a restricted list of values")

        self.info_lm_metrics = InfoLMMetricsInternal(
            model_name=model_name,
            temperature=temperature,
            measure_to_use=measure_to_use,
            use_idf_weights=use_idf_weights,
            alpha=alpha,
            beta=beta,
            device=device,
        )

    def evaluate_example(self, hyp: str, ref: str) -> float:
        # NOTE: due to the lack of implementation in InfoLMScoreMetricsInternal
        # we have to call evaluate_batch to handle one example
        sample_batch = self.evaluate_batch(hyps=[hyp], refs=[ref])
        return sample_batch[0]

    def evaluate_batch(self, hyps: tp.List[str], refs: tp.List[str]) -> tp.List[float]:
        info_lm_args = {}
        if self.use_idf_weights:
            idf_hyps, idf_ref = self.info_lm_metrics.prepare_idfs(hyps, refs)
            info_lm_args = {"idf_hyps": idf_hyps, "idf_ref": idf_ref}
        metrics_dict = self.info_lm_metrics.evaluate_batch(batch_hyps=hyps, batch_refs=refs, **info_lm_args)
        return metrics_dict[self.measure_to_use]

    def __repr__(self) -> str:
        return "InfoLM"
