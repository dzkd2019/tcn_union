"""TCN 集 + 手工特征 Ridge 混合评估的单元测试（CPU 合成数据）。

覆盖 57 维特征布局、Ridge 训练/预测的确定性、融合算术，以及
``ridge_weight=0`` 时融合结果等价于纯 TCN 集平均；不依赖真实数据。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_union.evaluation.hybrid_evaluate import (
    _available_seeds,
    _fit_ridge,
    _fuse,
    _mean_predictions,
    _predict_ridge,
    _ridge_features,
    _tcn_predictions,
)


def _prefix(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """构造 51 行（10 秒）、三通道为正的合成响应前缀。"""
    rng = np.random.default_rng(seed)
    time = np.arange(51, dtype=np.float64) * 0.2
    response = rng.uniform(1.0, 5.0, size=(51, 3))
    return time, response


class RidgeFeatureTest(unittest.TestCase):
    def test_features_have_57_dimensions_with_fixed_layout(self) -> None:
        time, response = _prefix(0)
        features = _ridge_features(time, response)
        self.assertEqual(features.shape, (57,))
        self.assertTrue(np.isfinite(features).all())
        # 三通道拼接，每通道 19 维；各通道第 0 维是 t=0 的绝对响应。
        np.testing.assert_allclose(
            features[[0, 19, 38]], response[0], rtol=0.0, atol=0.0
        )
        values = response[:, 0]
        absolute_index = [0, 5, 10, 25, 50]
        delta_index = [10, 25, 50]
        lag_index = [5, 20, 45]
        absolute = values[absolute_index]
        delta = values[delta_index] - values[0]
        np.testing.assert_allclose(features[0:5], absolute, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            features[5:10], np.log(np.maximum(absolute, 1e-6)), rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(features[10:13], delta, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            features[13:16], delta / np.abs(values[0]), rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            features[16:19],
            delta / np.maximum(np.abs(values[lag_index]), 1e-6),
            rtol=0.0,
            atol=0.0,
        )

    def test_features_require_full_10_second_prefix(self) -> None:
        rng = np.random.default_rng(1)
        time = np.arange(20, dtype=np.float64) * 0.2
        response = rng.uniform(1.0, 5.0, size=(20, 3))
        with self.assertRaises(ValueError):
            _ridge_features(time, response)


class RidgeRegressionTest(unittest.TestCase):
    def test_fit_predict_is_deterministic_and_positive(self) -> None:
        rng = np.random.default_rng(2)
        x_train = rng.normal(size=(50, 57))
        # 正浓度目标（log 空间近似线性），保证预测经 exp 后恒为正。
        y_train = np.exp(2.0 + 0.3 * x_train[:, :3] + 0.1 * rng.normal(size=(50, 3)))
        x_test = rng.normal(size=(12, 57))
        first = _predict_ridge(_fit_ridge(x_train, y_train), x_test)
        second = _predict_ridge(_fit_ridge(x_train, y_train), x_test)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (12, 3))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first > 0).all())


class FusionTest(unittest.TestCase):
    def test_fuse_weight_arithmetic(self) -> None:
        tcn = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ridge = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        np.testing.assert_allclose(_fuse(tcn, ridge, 0.0), tcn)
        np.testing.assert_allclose(_fuse(tcn, ridge, 1.0), ridge)
        np.testing.assert_allclose(_fuse(tcn, ridge, 0.5), (tcn + ridge) / 2.0)

    def test_ridge_weight_zero_equals_tcn_ensemble_mean(self) -> None:
        rng = np.random.default_rng(3)
        seed_predictions = [rng.uniform(1.0, 10.0, size=(5, 3)) for _ in range(3)]
        tcn_ensemble = _mean_predictions(seed_predictions)
        np.testing.assert_allclose(
            tcn_ensemble, np.mean(np.stack(seed_predictions), axis=0)
        )
        ridge_prediction = rng.uniform(1.0, 10.0, size=(5, 3))
        fused = _fuse(tcn_ensemble, ridge_prediction, 0.0)
        np.testing.assert_array_equal(fused, tcn_ensemble)


class CheckpointLoadingTest(unittest.TestCase):
    def test_missing_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _tcn_predictions(
                    [Path(tmp)], [], 10.0, [20260923], torch.device("cpu")
                )

    def test_available_seeds_supports_per_seed_directories(self) -> None:
        # 最终重训可每个目录只放一个 seed；扫描时按目录实际存在的请求 seed 收集。
        with tempfile.TemporaryDirectory() as tmp:
            window_dir = Path(tmp) / "10s"
            window_dir.mkdir()
            (window_dir / "seed_20260923.pt").touch()
            self.assertEqual(
                _available_seeds(Path(tmp), 10.0, [20260923, 20260924]),
                [20260923],
            )
            self.assertEqual(_available_seeds(Path(tmp), 10.0, [20260924]), [])


if __name__ == "__main__":
    unittest.main()
