from __future__ import annotations

import unittest

import torch

from tcn_union.core.model import GasTCN


class AuxiliaryPositionTest(unittest.TestCase):
    def test_last_position_matches_forward(self) -> None:
        torch.manual_seed(0)
        model = GasTCN(6).eval()
        x = torch.randn(4, 6, 51)
        with torch.no_grad():
            expected = model(x)
            actual = model.forward_positions(x, (50,))[0]
        torch.testing.assert_close(actual, expected)

    def test_early_position_is_causal(self) -> None:
        torch.manual_seed(0)
        model = GasTCN(6).eval()
        x = torch.randn(2, 6, 51)
        changed = x.clone()
        changed[:, :, 26:] = torch.randn(2, 6, 25) * 100.0
        with torch.no_grad():
            first = model.forward_positions(x, (25,))[0]
            second = model.forward_positions(changed, (25,))[0]
        torch.testing.assert_close(first, second)

    def test_multiple_positions_have_expected_shape(self) -> None:
        model = GasTCN(6).eval()
        x = torch.randn(3, 6, 51)
        with torch.no_grad():
            outputs = model.forward_positions(x, (25, 37))
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (3, 3))

    def test_invalid_position_rejected(self) -> None:
        model = GasTCN(6)
        with self.assertRaises(ValueError):
            model.forward_positions(torch.randn(1, 6, 51), (51,))


if __name__ == "__main__":
    unittest.main()
