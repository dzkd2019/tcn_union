"""GasTCN 逐气体输出头与 checkpoint head 协议测试（CPU 合成数据）。

覆盖：共享头/逐气体头的输出形状与参数字典键、forward_positions 一致性；
默认 `per_gas_head=False` 与显式 False 逐位一致；checkpoint 的
`architecture["head"]` 驱动 predict/_tcn_predict 重建；旧 checkpoint 缺该键
时按 shared 处理。
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
from tcn_union.training.train import _save_checkpoint, _train_network


def _identity_stats(in_channels: int) -> dict[str, np.ndarray]:
    """恒等 z-score 统计量，使推理期望可直接由模型输出得到。"""
    return {
        "x_mean": np.zeros(in_channels, dtype=np.float32),
        "x_std": np.ones(in_channels, dtype=np.float32),
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


class PerGasHeadModelTest(unittest.TestCase):
    def test_forward_shapes_for_both_head_modes(self) -> None:
        x = torch.randn(4, 6, 51)
        for per_gas_head in (False, True):
            model = GasTCN(6, per_gas_head=per_gas_head).eval()
            self.assertEqual(model.per_gas_head, per_gas_head)
            with torch.no_grad():
                output = model(x)
                positions = model.forward_positions(x, (25, 50))
            self.assertEqual(tuple(output.shape), (4, 3))
            self.assertEqual(
                [tuple(item.shape) for item in positions], [(4, 3), (4, 3)]
            )
            # 两种头在最后位置都必须与 forward 一致，早位置仍按因果卷积解码。
            torch.testing.assert_close(positions[-1], output)

    def test_state_dict_contains_expected_head_modules(self) -> None:
        shared_state = GasTCN(6, per_gas_head=False).state_dict()
        self.assertIn("head.0.weight", shared_state)
        self.assertIn("head.2.bias", shared_state)
        self.assertFalse(
            any(name.startswith("heads.") for name in shared_state),
            "shared head must not register per-gas heads",
        )
        per_gas = GasTCN(6, per_gas_head=True)
        per_gas_state = per_gas.state_dict()
        for name in ("heads.0.0.weight", "heads.1.2.bias", "heads.2.0.weight"):
            self.assertIn(name, per_gas_state)
        self.assertNotIn("head.0.weight", per_gas_state)
        for head in per_gas.heads:
            # 逐气体头的输入是 TCN 主干固定隐藏宽度 16，输出单气体。
            self.assertEqual(tuple(head[0].weight.shape), (16, 16))
            self.assertEqual(tuple(head[2].weight.shape), (1, 16))

    def test_per_gas_output_equals_concatenated_head_outputs(self) -> None:
        model = GasTCN(6, per_gas_head=True).eval()
        x = torch.randn(2, 6, 51)
        with torch.no_grad():
            hidden = model.blocks(x)
            expected = torch.cat(
                [head(hidden[:, :, -1]) for head in model.heads], dim=1
            )
            torch.testing.assert_close(model(x), expected)

    def test_default_matches_explicit_shared_head(self) -> None:
        torch.manual_seed(7)
        default = GasTCN(6)
        torch.manual_seed(7)
        explicit = GasTCN(6, per_gas_head=False)
        default_state = default.state_dict()
        explicit_state = explicit.state_dict()
        self.assertEqual(list(default_state), list(explicit_state))
        for name, value in default_state.items():
            self.assertTrue(torch.equal(value, explicit_state[name]), name)
        x = torch.randn(3, 6, 51)
        default.eval()
        explicit.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                default(x), explicit(x), rtol=0.0, atol=0.0
            )

    def test_train_network_builds_per_gas_head(self) -> None:
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
            per_gas_head=True,
        )
        self.assertEqual(epochs, 1)
        self.assertTrue(model.per_gas_head)
        self.assertIn("heads.2.2.weight", model.state_dict())
        model.eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.from_numpy(x)).shape), (8, 3))


class CheckpointHeadProtocolTest(unittest.TestCase):
    def _save(self, path: Path, per_gas_head: bool) -> GasTCN:
        """保存共享头/逐气体头的手工 checkpoint；统计量为恒等变换。"""
        torch.manual_seed(11)
        model = GasTCN(6, per_gas_head=per_gas_head)
        _save_checkpoint(
            path,
            model,
            _identity_stats(6),
            "raw_delta",
            10.0,
            [{"file": "synthetic.csv", "sha256": "0" * 64}],
            123,
            1,
            torch.device("cpu"),
            per_gas_head=per_gas_head,
        )
        return model

    def test_architecture_records_shared_head_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            self._save(path, per_gas_head=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["architecture"]["head"], "shared")

    def test_architecture_records_per_gas_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            self._save(path, per_gas_head=True)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["architecture"]["head"], "per_gas")

    def test_per_gas_checkpoint_drives_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "seed.pt"
            model = self._save(checkpoint_path, per_gas_head=True)
            csv_path = tmp_path / "sample.csv"
            time, responses = _write_prefix_csv(csv_path)
            features = make_features(time, responses, 10.0, "raw_delta")
            expected = _raw_logits(model, features[None])[0]
            result = predict(checkpoint_path, csv_path, torch.device("cpu"))
        np.testing.assert_allclose(
            [result["H2"], result["NH3"], result["NO2"]],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_per_gas_checkpoint_drives_tcn_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "seed.pt"
            model = self._save(checkpoint_path, per_gas_head=True)
            features = np.random.default_rng(8).normal(size=(3, 6, 51)).astype(
                np.float32
            )
            prediction, checkpoint = _tcn_predict(
                checkpoint_path, features, torch.device("cpu")
            )
        self.assertEqual(checkpoint["architecture"]["head"], "per_gas")
        np.testing.assert_allclose(
            prediction, _raw_logits(model, features), rtol=1e-6, atol=1e-6
        )

    def test_shared_checkpoint_drives_tcn_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "seed.pt"
            model = self._save(checkpoint_path, per_gas_head=False)
            features = np.random.default_rng(9).normal(size=(2, 6, 51)).astype(
                np.float32
            )
            prediction, checkpoint = _tcn_predict(
                checkpoint_path, features, torch.device("cpu")
            )
        self.assertEqual(checkpoint["architecture"]["head"], "shared")
        np.testing.assert_allclose(
            prediction, _raw_logits(model, features), rtol=1e-6, atol=1e-6
        )

    def test_old_checkpoint_without_head_key_defaults_to_shared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.pt"
            model = self._save(path, per_gas_head=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            # 模拟旧 checkpoint：architecture 缺 head 键时应按 shared 重建。
            del payload["architecture"]["head"]
            torch.save(payload, path)
            features = np.random.default_rng(4).normal(size=(2, 6, 51)).astype(
                np.float32
            )
            prediction, checkpoint = _tcn_predict(
                path, features, torch.device("cpu")
            )
        self.assertNotIn("head", checkpoint["architecture"])
        np.testing.assert_allclose(
            prediction, _raw_logits(model, features), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
