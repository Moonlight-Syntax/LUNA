import typing as tp

import pytest

from luna.statistical_measures import BaryScoreMetrics, DepthScoreMetrics, InfoLMMetrics
from tests.conftest import get_mock_data

HYPS, REFS = get_mock_data()
ABS = 1e-1


class TestBaryScore:
    DEFAULT_BARY_SCORE_PARAMS = {
        "model_name": "bert-base-uncased",
        "last_layers": 5,
        "use_idfs": True,
        "sinkhorn_ref": 0.01,
    }


    @pytest.mark.parametrize(
        ["ind", "gt_value"],
        (
            (1, 0.4650),
            (2, 0.3796),
        ),
     )
    def test_evaluate_example(self, ind: int, gt_value: float) -> None:
        metrics = BaryScoreMetrics(**self.DEFAULT_BARY_SCORE_PARAMS)
        hyp, ref = HYPS[ind], REFS[ind]

        value = metrics.evaluate_example(hyp, ref)
        assert value == pytest.approx(gt_value, abs=ABS)

    @pytest.mark.parametrize(
        ["indices", "min_value", "max_value"],
        (
            ([0, 2, 3, 5], 0.0, 0.4100),
            ([1, 4], 0.4651, 0.6205),
        ),
    )
    def test_evaluate_batch(self, indices: tp.List[int], min_value: float, max_value: float) -> None:
        metrics = BaryScoreMetrics(**self.DEFAULT_BARY_SCORE_PARAMS)

        hyps = [HYPS[ind] for ind in indices]
        refs = [REFS[ind] for ind in indices]
        metrics_list = metrics.evaluate_batch(hyps=hyps, refs=refs)

        assert len(metrics_list) == len(indices)
        assert min(metrics_list) == pytest.approx(min_value, abs=ABS) and max(metrics_list) == pytest.approx(max_value, abs=ABS)


    @pytest.mark.parametrize(
        ["bary_score_params", "min_value", "max_value"],
        (
            ({"last_layers": 1, "use_idfs": False}, 0.0, 0.7754),
            ({"last_layers": 3, "sinkhorn_ref": 0.5}, 0.1666, 0.6506),
            ({"model_name": "bert-base-cased"}, 0.00, 0.4149),
        ),
    )
    def test_multiple_bary_score_params(self, bary_score_params: tp.Dict[str, tp.Any], min_value: float, max_value: float) -> None:
        params = self.DEFAULT_BARY_SCORE_PARAMS.copy()
        params.update(bary_score_params)
        metrics = BaryScoreMetrics(**params)

        metrics_list = metrics.evaluate_batch(hyps=HYPS, refs=REFS)

        assert len(metrics_list) == len(HYPS)
        assert min(metrics_list) == pytest.approx(min_value, abs=ABS) and max(metrics_list) == pytest.approx(max_value, abs=ABS)


class TestDepthScore:
    DEFAULT_DEPTH_SCORE_PARAMS = {
        "model_name": "bert-base-uncased",
        "layers_to_consider": 9,
        "considered_measure": "irw"
    }

    @pytest.mark.parametrize(
        ["ind", "gt_value"],
        (
            (1, 0.1375),
            (2, 0.1015),
        ),
    )
    def test_evaluate_example(self, ind: int, gt_value: float) -> None:
        metrics = DepthScoreMetrics(**self.DEFAULT_DEPTH_SCORE_PARAMS)
        hyp, ref = HYPS[ind], REFS[ind]

        value = metrics.evaluate_example(hyp, ref)
        assert value == pytest.approx(gt_value, abs=ABS)


    @pytest.mark.parametrize(
        ["indices", "min_value", "max_value"],
        (
            ([0, 2, 3, 5], 0.0, 0.1015),
            ([1, 4], 0.1375, 0.1399),
        ),
    )
    def test_evaluate_batch(self, indices: tp.List[int], min_value: float, max_value: float) -> None:
        metrics = DepthScoreMetrics(**self.DEFAULT_DEPTH_SCORE_PARAMS)

        hyps = [HYPS[ind] for ind in indices]
        refs = [REFS[ind] for ind in indices]
        metrics_list = metrics.evaluate_batch(hyps=hyps, refs=refs)

        assert len(metrics_list) == len(indices)
        assert min(metrics_list) == pytest.approx(min_value, abs=ABS) and max(metrics_list) == pytest.approx(max_value, abs=ABS)

    @pytest.mark.parametrize(
        ["depth_score_params", "min_value", "max_value"],
        (
            ({"layers_to_consider": 3, "considered_measure": "mmd"}, 0.00, 0.1324),
            ({"layers_to_consider": 6, "considered_measure": "wasserstein"}, 0.00, 0.8087),
            ({"layers_to_consider": 1, "considered_measure": "sliced"}, 0.00, 0.02),
        ),
    )
    def test_multiple_depth_score_params(self, depth_score_params: tp.Dict[str, tp.Any], min_value: float, max_value: float) -> None:
        params = self.DEFAULT_DEPTH_SCORE_PARAMS.copy()
        params.update(depth_score_params)
        metrics = DepthScoreMetrics(**params)

        metrics_list = metrics.evaluate_batch(hyps=HYPS, refs=REFS)

        assert len(metrics_list) == len(HYPS)
        assert min(metrics_list) == pytest.approx(min_value, abs=ABS) and max(metrics_list) == pytest.approx(max_value, abs=ABS)


class TestInfoLM:
    DEFAULT_INFO_LM_PARAMS = {
        "model_name": "bert-base-uncased",
        "temperature": 0.25,
        "measure_to_use": "fisher_rao",
        "use_idf_weights": True,
        "alpha": None,
        "beta": None,
    }

    @pytest.mark.parametrize(
        ["ind", "gt_value"],
        (
            (1, 2.7269),
            (2, 1.6808),
        ),
    )
    def test_evaluate_example(self, ind: int, gt_value: float) -> None:
        # NOTE: we use use_idf_weights=False for 1 example test.
        # Otherwise, metrics value will be equal zero.

        params = self.DEFAULT_INFO_LM_PARAMS.copy()
        params.update({"use_idf_weights": False})

        metrics = InfoLMMetrics(**params)
        hyp, ref = HYPS[ind], REFS[ind]

        value = metrics.evaluate_example(hyp, ref)
        assert value == pytest.approx(gt_value, abs=ABS)

    @pytest.mark.parametrize(
        ["indices", "min_value", "max_value"],
        (
            ([0, 2, 3, 5], 0.0, 2.430),
            ([1, 4], 2.6616, 2.7733),
        ),
    )
    def test_evaluate_batch(self, indices: tp.List[int], min_value: float, max_value: float) -> None:
        metrics = InfoLMMetrics(**self.DEFAULT_INFO_LM_PARAMS)

        hyps = [HYPS[ind] for ind in indices]
        refs = [REFS[ind] for ind in indices]
        metrics_list = metrics.evaluate_batch(hyps=hyps, refs=refs)

        assert len(metrics_list) == len(indices)
        assert min(metrics_list) == pytest.approx(min_value, abs=ABS) and max(metrics_list) == pytest.approx(max_value, abs=ABS)

    @pytest.mark.parametrize(
        ["info_lm_params", "min_value", "max_value"],
        (
            ({"temperature": 1.0, "use_idf_weights": False}, 0.00, 2.5382),
            ({"measure_to_use": "ab", "alpha": 0.25, "beta": 2.0}, 0.00, 3.2985),
            ({"measure_to_use": "renyi", "alpha": 1.5, "use_idf_weights": False}, 0.00, 54.999),
        ),
    )
    def test_multiple_info_lm_params(self, info_lm_params: tp.Dict[str, tp.Any], min_value: float, max_value: float) -> None:
        params = self.DEFAULT_INFO_LM_PARAMS.copy()
        params.update(info_lm_params)
        metrics = InfoLMMetrics(**params)

        metrics_list = metrics.evaluate_batch(hyps=HYPS, refs=REFS)

        assert len(metrics_list) == len(HYPS)
        assert min(metrics_list) == pytest.approx(min_value, abs=ABS) and max(metrics_list) == pytest.approx(max_value, abs=ABS)
