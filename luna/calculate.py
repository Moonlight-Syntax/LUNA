import typing as tp

import joblib

from luna.base import Metrics
from luna.reference_free import ReferenceFreeMetrics


class Calculator:
    """
    Attributes
    ----------
    execute_parallel: bool
        If True, apply joblib.Parallel mechanism across metrics.
    """

    def __init__(self, execute_parallel: bool = False) -> None:
        self.execute_parallel = execute_parallel

    def _get_kwargs(self, metrics_impl: Metrics, hyps: tp.List[str], refs: tp.List[str]) -> tp.Dict[str, tp.Any]:
        if isinstance(metrics_impl, ReferenceFreeMetrics):
            return {"input_texts": refs, "hyps": hyps}
        return {"hyps": hyps, "refs": refs}

    def _execute_sequentially(
        self, metrics: tp.List[Metrics], hyps: tp.List[str], refs: tp.List[str]
    ) -> tp.Dict[str, tp.Union[int, float]]:
        result = {}
        for metrics_impl in metrics:
            key = str(metrics_impl)
            evaluate_kwargs = self._get_kwargs(metrics_impl, hyps, refs)
            result[key] = metrics_impl.evaluate_batch(**evaluate_kwargs)
        return result

    def _execute_parallel(
        self, metrics_impl: Metrics, hyps: tp.List[str], refs: tp.List[str]
    ) -> tp.Tuple[str, tp.Union[int, float]]:
        evaluate_kwargs = self._get_kwargs(metrics_impl, hyps, refs)
        value = metrics_impl.evaluate_batch(**evaluate_kwargs)
        return str(metrics_impl), value

    def calculate(
        self, metrics: tp.List[Metrics], hyps: tp.List[str], refs: tp.List[str]
    ) -> tp.Dict[str, tp.Union[int, float]]:
        if not metrics:
            raise RuntimeError("Pass the positive number of metrics")
        for metrics_impl in metrics:
            if not isinstance(metrics_impl, Metrics):
                raise TypeError("Each instance of the metrics list should be inherited from base.Metrics")

        if self.execute_parallel:
            _parallel = joblib.Parallel(n_jobs=len(metrics))
            output_generator = _parallel(
                joblib.delayed(self._execute_parallel)(metrics_impl, hyps, refs) for metrics_impl in metrics
            )
            result = {tup[0]: tup[1] for tup in output_generator}
            return result
        return self._execute_sequentially(metrics, hyps, refs)
