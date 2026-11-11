"""可配置宽度 GasTCN 与新特征模式 raw_delta_rel 的契约测试（CPU 合成数据）。

不依赖真实数据训练：宽度/通道顺序用随机张量检查；checkpoint 读写用恒等
标准化统计量构造手工 checkpoint，覆盖 predict 与 evaluate 两条推理链路。
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
from tcn_union.training.train import _channel_order, _save_checkpoint, _train_network


def _identity_stats(channels: int) -> dict[str, np.ndarray]:
    """恒等 z-score 统计量，使推理期望可直接由模型输出得到。"""
    return {
        "x_mean": np.zeros(channels, dtype=np.float32),
        "x_std": np.ones(channels, dtype=np.float32),
        "y_mean": np.zeros(3, dtype=np.float32),
        "y_std": np.ones(3, dtype=np.float32),
    }


def _raw_logits(model: GasTCN, features: np.ndarray) -> np.ndarray:
    """在 eval 模式下复算模型输出，作为推理链路的独立期望。"""
    model.eval()
    with torch.no_grad():
        return np.stack(
            [
                model(torch.from_numpy(sample).unsqueeze(0)).numpy()[0]
                for sample in features
            ]
        )


def _write_prefix_csv(path: Path, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """写一份 10 秒合成前缀 CSV；初值取正数，避免相对增量被 1e-8 放大。"""
    rng = np.random.default_rng(seed)
    time = np.arange(51, dtype=np.float64) * 0.2
    responses = rng.uniform(1.0, 10.0, size=(51, 3))
    np.savetxt(
        path,
        np.column_stack((time, responses)),
        delimiter=",",
        header="time,H2,NH3,NO2",
        comments="",
    )
    return time, responses


class GasTCNWidthTest(unittest.TestCase):
    def test_default_width_matches_explicit_sixteen(self) -> None:
        torch.manual_seed(7)
        default = GasTCN(6)
        torch.manual_seed(7)
        explicit = GasTCN(6, channels=16)
        self.assertEqual(default.channels, 16)
        default_state = default.state_dict()
        explicit_state = explicit.state_dict()
        self.assertEqual(list(default_state), list(explicit_state))
        for name, value in default_state.items():
            self.assertTrue(torch.equal(value, explicit_state[name]), name)
        x = torch.randn(4, 6, 51)
        default.eval()
        explicit.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                default(x), explicit(x), rtol=0.0, atol=0.0
            )

    def test_narrow_and_wide_forward_shapes(self) -> None:
        for in_channels, channels in ((6, 8), (9, 32)):
            model = GasTCN(in_channels, channels=channels).eval()
            self.assertEqual(
                (model.in_channels, model.channels), (in_channels, channels)
            )
            x = torch.randn(3, in_channels, 51)
            with torch.no_grad():
                output = model(x)
                positions = model.forward_positions(x, (25, 50))
            self.assertEqual(tuple(output.shape), (3, 3))
            self.assertEqual([tuple(item.shape) for item in positions], [(3, 3), (3, 3)])
            # 最后位置与 forward 必须一致，早位置仍按因果卷积解码。
            torch.testing.assert_close(positions[-1], output)

    def test_hidden_layer_shapes_follow_channels(self) -> None:
        model = GasTCN(9, channels=32)
        self.assertEqual(tuple(model.blocks[0].conv1.weight.shape), (32, 9, 5))
        self.assertEqual(tuple(model.blocks[0].projection.weight.shape), (32, 9, 1))
        for block in list(model.blocks)[1:]:
            self.assertEqual(tuple(block.conv1.weight.shape), (32, 32, 5))
            self.assertEqual(tuple(block.conv2.weight.shape), (32, 32, 5))
            self.assertIsInstance(block.projection, torch.nn.Identity)
        self.assertEqual(tuple(model.head[0].weight.shape), (16, 32))
        self.assertEqual(tuple(model.head[2].weight.shape), (3, 16))

    def test_invalid_input_channels_rejected(self) -> None:
        model = GasTCN(6, channels=8)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 9, 51))
        with self.assertRaises(ValueError):
            model.forward_positions(torch.randn(1, 9, 51), (50,))

    def test_train_network_uses_requested_channels(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(8, 6, 51)).astype(np.float32)
        y = rng.normal(size=(8, 3)).astype(np.float32)
        model, _, epochs, _, _ = _train_network(
            x,
            y,
            None,
            None,
            y,
            1,
            123,
            torch.device("cpu"),
            False,
            channels=8,
        )
        self.assertEqual(epochs, 1)
        self.assertEqual(model.channels, 8)
        model.eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.from_numpy(x)).shape), (8, 3))


class ChannelOrderTest(unittest.TestCase):
    def test_legacy_orders_unchanged(self) -> None:
        self.assertEqual(
            _channel_order("raw_delta"),
            ["H2_raw", "NH3_raw", "NO2_raw", "H2_delta", "NH3_delta", "NO2_delta"],
        )
        self.assertEqual(
            _channel_order("delta"), ["H2_delta", "NH3_delta", "NO2_delta"]
        )

    def test_raw_delta_rel_order(self) -> None:
        self.assertEqual(
            _channel_order("raw_delta_rel"),
            [
                "H2_raw",
                "NH3_raw",
                "NO2_raw",
                "H2_delta_abs",
                "NH3_delta_abs",
                "NO2_delta_abs",
                "H2_rel_delta",
                "NH3_rel_delta",
                "NO2_rel_delta",
            ],
        )


class CheckpointChannelsTest(unittest.TestCase):
    def _save(self, path: Path, mode: str, in_channels: int, channels: int) -> GasTCN:
        """保存可选宽度的手工 checkpoint；统计量为恒等变换。"""
        torch.manual_seed(11)
        model = GasTCN(in_channels, channels=channels)
        _save_checkpoint(
            path,
            model,
            _identity_stats(in_channels),
            mode,
            10.0,
            [{"file": "synthetic.csv", "sha256": "0" * 64}],
            123,
            1,
            torch.device("cpu"),
            channels=channels,
        )
        return model

    def test_save_checkpoint_records_architecture_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            self._save(path, "raw_delta_rel", 9, 32)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["architecture"]["in_channels"], 9)
        self.assertEqual(payload["architecture"]["channels"], 32)
        self.assertIn("relative_delta", payload["architecture"])
        self.assertEqual(payload["mode"], "raw_delta_rel")
        self.assertEqual(len(payload["channel_order"]), 9)

    def test_default_checkpoint_keeps_sixteen_and_no_relative_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            self._save(path, "raw_delta", 6, 16)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["architecture"]["channels"], 16)
        self.assertNotIn("relative_delta", payload["architecture"])
        self.assertEqual(len(payload["channel_order"]), 6)

    def test_predict_reads_wide_raw_delta_rel_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "seed.pt"
            model = self._save(checkpoint_path, "raw_delta_rel", 9, 32)
            csv_path = tmp_path / "sample.csv"
            time, responses = _write_prefix_csv(csv_path)
            features = make_features(time, responses, 10.0, "raw_delta_rel")
            expected = _raw_logits(model, features[None])[0]
            result = predict(checkpoint_path, csv_path, torch.device("cpu"))
        self.assertEqual(result["seconds"], 10.0)
        np.testing.assert_allclose(
            [result["H2"], result["NH3"], result["NO2"]],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_tcn_predict_reads_checkpoint_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "seed.pt"
            model = self._save(checkpoint_path, "raw_delta_rel", 9, 32)
            features = np.random.default_rng(8).normal(size=(3, 9, 51)).astype(
                np.float32
            )
            prediction, checkpoint = _tcn_predict(
                checkpoint_path, features, torch.device("cpu")
            )
        self.assertEqual(checkpoint["architecture"]["channels"], 32)
        np.testing.assert_allclose(
            prediction, _raw_logits(model, features), rtol=1e-6, atol=1e-6
        )

    def test_old_checkpoint_without_channels_defaults_to_sixteen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            model = self._save(path, "raw_delta", 6, 16)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            # 模拟旧 checkpoint：architecture 缺 channels 键时应按 16 重建网络。
            del payload["architecture"]["channels"]
            torch.save(payload, path)
            features = np.random.default_rng(4).normal(size=(2, 6, 51)).astype(
                np.float32
            )
            prediction, checkpoint = _tcn_predict(
                path, features, torch.device("cpu")
            )
        self.assertNotIn("channels", checkpoint["architecture"])
        np.testing.assert_allclose(
            prediction, _raw_logits(model, features), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
