from __future__ import annotations

import unittest

import torch

from tcn_union.core.device import configure_threads, describe_device, resolve_device


class DeviceTest(unittest.TestCase):
    def test_cpu_explicit(self) -> None:
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")
        configure_threads(device)
        self.assertEqual(describe_device(device), "CPU")

    def test_auto_returns_cpu_or_gpu(self) -> None:
        self.assertIn(resolve_device("auto").type, {"cpu", "cuda"})

    def test_invalid_device_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    @unittest.skipIf(torch.cuda.is_available(), "GPU is available")
    def test_cuda_rejected_when_unavailable(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_device("cuda")


if __name__ == "__main__":
    unittest.main()
