"""目标变换的训练、checkpoint 与推理还原测试（CPU 合成数据）。

不依赖仓库中的真实数据：训练路径用正浓度合成标签，推理路径用手工 checkpoint
和合成 CSV，覆盖单值与逐气体（如 none,log,log）配置在训练、保存和
predict/evaluate 两条推理链路上的还原。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_union.cli.predict import predict
from tcn_union.core.data import make_features
from tcn_union.core.model import GasTCN
from tcn_union.evaluation.evaluate import _tcn_predict
from tcn_union.training.train import LOG_EPS, _save_checkpoint, _train_network

# 手工 checkpoint 的 log 空间标签统计量，模拟 H2/NH3/NO2 的浓度量级。
LOG_MEAN = np.array([np.log(16000.0), np.log(30.0), np.log(6.0)], dtype=np.float32)
LOG_STD = np.array([0.4, 0.5, 0.3], dtype=np.float32)


def _synthetic(seed: int, runs: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """构造正浓度合成数据；标签跨约一个数量级，使 log/线性空间差异明显。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(runs, 6, 51)).astype(np.float32)
    y = np.exp(rng.uniform(np.log(2.0), np.log(50.0), size=(runs, 3))).astype(
        np.float32
    )
    return x, y


def _save_manual_checkpoint(
    path: Path, target_transform: str | list[str] = "log"
) -> GasTCN:
    """用恒等输入标准化保存一个已知模型的 checkpoint，便于手算推理期望。"""
    torch.manual_seed(0)
    model = GasTCN(6)
    stats = {
        "x_mean": np.zeros(6, dtype=np.float32),
        "x_std": np.ones(6, dtype=np.float32),
        "y_mean": LOG_MEAN.copy(),
        "y_std": LOG_STD.copy(),
    }
    _save_checkpoint(
        path,
        model,
        stats,
        "raw_delta",
        10.0,
        [{"file": "synthetic.csv", "sha256": "0" * 64}],
        123,
        1,
        torch.device("cpu"),
        target_transform=target_transform,
    )
    return model


def _raw_logits(model: GasTCN, features: np.ndarray) -> np.ndarray:
    """在 eval 模式下复算模型的标准化输出，作为推理还原的独立期望。"""
    model.eval()
    with torch.no_grad():
        return np.stack(
            [
                model(torch.from_numpy(sample).unsqueeze(0)).numpy()[0]
                for sample in features
            ]
        )


def _write_prefix_csv(
    tmp_path: Path, seed: int
) -> tuple[Path, np.ndarray, np.ndarray]:
    """写一份 10 秒合成前缀 CSV，返回路径与构造特征用的时间/响应。"""
    rng = np.random.default_rng(seed)
    time = np.arange(51, dtype=np.float64) * 0.2
    responses = rng.normal(size=(51, 3))
    csv_path = tmp_path / "sample.csv"
    np.savetxt(
        csv_path,
        np.column_stack((time, responses)),
        delimiter=",",
        header="time,H2,NH3,NO2",
        comments="",
    )
    return csv_path, time, responses


class LogTargetTrainingTest(unittest.TestCase):
    def test_log_training_uses_log_space_and_returns_positive_predictions(
        self,
    ) -> None:
        x, y = _synthetic(0)
        model, stats, epochs, score, prediction = _train_network(
            x,
            y,
            x,
            y,
            y,
            2,
            123,
            torch.device("cpu"),
            True,
            target_transform="log",
        )
        self.assertEqual(epochs, 2)
        self.assertIsNotNone(score)
        self.assertIsNotNone(prediction)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())
        # 标签 z-score 统计量应拟合在 log 空间，而不是原始浓度空间。
        expected_mean = np.log(np.maximum(y.astype(np.float64), LOG_EPS)).mean(axis=0)
        expected_std = np.log(np.maximum(y.astype(np.float64), LOG_EPS)).std(axis=0)
        self.assertTrue(
            np.allclose(stats["y_mean"], expected_mean, rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(
            np.allclose(stats["y_std"], expected_std, rtol=1e-5, atol=1e-5)
        )
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all())

    def test_per_gas_training_mixes_log_and_raw_targets(self) -> None:
        x, y = _synthetic(6)
        model, stats, epochs, score, prediction = _train_network(
            x,
            y,
            x,
            y,
            y,
            2,
            123,
            torch.device("cpu"),
            True,
            target_transform="none,log,log",
        )
        self.assertEqual(epochs, 2)
        self.assertIsNotNone(score)
        self.assertIsNotNone(prediction)
        self.assertTrue(np.isfinite(prediction).all())
        # NH3/NO2 在 log 空间训练，还原后必为正；H2 保持原始浓度空间，在当前
        # 固定合成数据下也为正，一并断言避免还原逻辑产生异常符号。
        self.assertTrue((prediction > 0).all())
        self.assertTrue((prediction[:, 1:] > 0).all())
        raw = y.astype(np.float64)
        expected_mean = np.stack(
            (raw[:, 0], np.log(raw[:, 1]), np.log(raw[:, 2])), axis=1
        ).mean(axis=0)
        self.assertTrue(
            np.allclose(stats["y_mean"], expected_mean, rtol=1e-5, atol=1e-5)
        )
        # 逐气体配置确实生效：统计量不等于三气体都在原始空间的结果。
        self.assertFalse(
            np.allclose(stats["y_mean"], raw.mean(axis=0), rtol=1e-5, atol=1e-5)
        )
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all())

    def test_invalid_target_transform_rejected(self) -> None:
        x, y = _synthetic(1, runs=4)
        with self.assertRaises(ValueError):
            _train_network(
                x,
                y,
                None,
                None,
                y,
                1,
                123,
                torch.device("cpu"),
                False,
                target_transform="sqrt",
            )

    def test_invalid_per_gas_specs_rejected(self) -> None:
        x, y = _synthetic(7, runs=4)
        # 非法配置：长度不为 3（空串/两值/四值）、未知气体空间、大小写错误。
        for spec in ("", "none,log", "none,log,log,log", "none,log,sqrt", "NONE,log,log"):
            with self.assertRaises(ValueError, msg=spec):
                _train_network(
                    x,
                    y,
                    None,
                    None,
                    y,
                    1,
                    123,
                    torch.device("cpu"),
                    False,
                    target_transform=spec,
                )

    def test_default_none_matches_explicit_none(self) -> None:
        x, y = _synthetic(2)
        default = _train_network(
            x, y, None, None, y, 2, 123, torch.device("cpu"), False
        )
        explicit = _train_network(
            x,
            y,
            None,
            None,
            y,
            2,
            123,
            torch.device("cpu"),
            False,
            target_transform="none",
        )
        # 同一 seed 下默认路径与显式 none 必须逐位一致，即旧行为不变。
        for (name, value), (_, other) in zip(
            default[0].state_dict().items(), explicit[0].state_dict().items()
        ):
            self.assertTrue(torch.equal(value, other), name)
        for key in ("x_mean", "x_std", "y_mean", "y_std"):
            self.assertTrue(np.array_equal(default[1][key], explicit[1][key]), key)
        # none 模式标签统计量仍在原始浓度空间。
        self.assertTrue(
            np.allclose(default[1]["y_mean"], y.mean(axis=0), rtol=1e-5, atol=1e-5)
        )


class LogTargetInferenceTest(unittest.TestCase):
    def test_log_checkpoint_restores_concentrations_with_exp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            model = _save_manual_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["target_transform"], "log")
            features = np.random.default_rng(3).normal(size=(2, 6, 51)).astype(
                np.float32
            )
            prediction, checkpoint = _tcn_predict(
                path, features, torch.device("cpu")
            )
            self.assertEqual(checkpoint["target_transform"], "log")
            expected = np.exp(_raw_logits(model, features) * LOG_STD + LOG_MEAN)
            np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-6)
            self.assertTrue((prediction > 0).all())

    def test_per_gas_checkpoint_restores_each_gas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            model = _save_manual_checkpoint(
                path, target_transform=["none", "log", "log"]
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["target_transform"], ["none", "log", "log"])
            features = np.random.default_rng(8).normal(size=(2, 6, 51)).astype(
                np.float32
            )
            prediction, checkpoint = _tcn_predict(
                path, features, torch.device("cpu")
            )
            self.assertEqual(
                checkpoint["target_transform"], ["none", "log", "log"]
            )
            raw = _raw_logits(model, features) * LOG_STD + LOG_MEAN
            expected = raw.copy()
            # H2 保持原始空间，NH3/NO2 逐气体取指数还原。
            expected[:, 1:] = np.exp(raw[:, 1:])
            np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-6)
            self.assertTrue((prediction[:, 1:] > 0).all())

    def test_uniform_per_gas_spec_stored_as_single_value(self) -> None:
        # 三气体一致时 checkpoint 仍写单值字符串，与旧推理协议/旧 checkpoint 兼容。
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "log.pt"
            _save_manual_checkpoint(log_path, target_transform="log,log,log")
            payload = torch.load(log_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["target_transform"], "log")
            none_path = Path(tmp) / "none.pt"
            _save_manual_checkpoint(none_path, target_transform="none")
            payload = torch.load(none_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["target_transform"], "none")

    def test_old_checkpoint_without_key_defaults_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            model = _save_manual_checkpoint(path)
            # 模拟旧 checkpoint：缺失 target_transform 键时应按 none 处理。
            payload = torch.load(path, map_location="cpu", weights_only=False)
            del payload["target_transform"]
            torch.save(payload, path)
            features = np.random.default_rng(4).normal(size=(2, 6, 51)).astype(
                np.float32
            )
            prediction, checkpoint = _tcn_predict(
                path, features, torch.device("cpu")
            )
            self.assertNotIn("target_transform", checkpoint)
            expected = _raw_logits(model, features) * LOG_STD + LOG_MEAN
            np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-6)

    def test_predict_restores_log_concentrations_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "seed.pt"
            model = _save_manual_checkpoint(checkpoint_path)
            csv_path, time, responses = _write_prefix_csv(tmp_path, 5)
            features = make_features(time, responses, 10.0, "raw_delta")
            expected = np.exp(
                _raw_logits(model, features[None, :, :]) * LOG_STD + LOG_MEAN
            )
            result = predict(checkpoint_path, csv_path, torch.device("cpu"))
            np.testing.assert_allclose(
                [result["H2"], result["NH3"], result["NO2"]],
                expected[0],
                rtol=1e-5,
                atol=1e-5,
            )
            self.assertEqual(result["unit"], "original")
            self.assertEqual(result["seconds"], 10.0)

    def test_predict_restores_per_gas_concentrations_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "seed.pt"
            model = _save_manual_checkpoint(
                checkpoint_path, target_transform=["none", "log", "log"]
            )
            csv_path, time, responses = _write_prefix_csv(tmp_path, 9)
            features = make_features(time, responses, 10.0, "raw_delta")
            raw = _raw_logits(model, features[None, :, :]) * LOG_STD + LOG_MEAN
            expected = raw[0].copy()
            expected[1:] = np.exp(expected[1:])
            result = predict(checkpoint_path, csv_path, torch.device("cpu"))
            np.testing.assert_allclose(
                [result["H2"], result["NH3"], result["NO2"]],
                expected,
                rtol=1e-5,
                atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
