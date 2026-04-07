from __future__ import annotations

import unittest

import torch

from src.models import PhoneWatchContrastiveModel, ProjectionHead, TimeSeriesEncoder, symmetric_info_nce_loss


class ContrastiveModelTests(unittest.TestCase):
    def test_encoder_and_projection_shapes(self) -> None:
        batch_size = 4
        sequence_length = 60
        x_phone = torch.randn(batch_size, 6, sequence_length)
        x_watch = torch.randn(batch_size, 6, sequence_length)

        model = PhoneWatchContrastiveModel()
        output = model(x_phone, x_watch)

        self.assertEqual(output.h_phone.shape, (batch_size, TimeSeriesEncoder.output_dim))
        self.assertEqual(output.h_watch.shape, (batch_size, TimeSeriesEncoder.output_dim))
        self.assertEqual(output.z_phone.shape, (batch_size, 64))
        self.assertEqual(output.z_watch.shape, (batch_size, 64))

        phone_norms = torch.linalg.vector_norm(output.z_phone, dim=1)
        watch_norms = torch.linalg.vector_norm(output.z_watch, dim=1)
        self.assertTrue(torch.allclose(phone_norms, torch.ones_like(phone_norms), atol=1e-5))
        self.assertTrue(torch.allclose(watch_norms, torch.ones_like(watch_norms), atol=1e-5))

    def test_projection_head_normalizes_output(self) -> None:
        head = ProjectionHead(in_dim=TimeSeriesEncoder.output_dim, hidden_dim=128, out_dim=64)
        x = torch.randn(3, TimeSeriesEncoder.output_dim)
        z = head(x)
        self.assertEqual(z.shape, (3, 64))
        norms = torch.linalg.vector_norm(z, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_symmetric_info_nce_returns_finite_loss(self) -> None:
        torch.manual_seed(42)
        z_phone = torch.randn(8, 64)
        z_watch = z_phone + 0.05 * torch.randn(8, 64)

        result = symmetric_info_nce_loss(z_phone, z_watch, temperature=0.2)

        self.assertEqual(result.similarity_matrix.shape, (8, 8))
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(torch.isfinite(result.phone_to_watch_loss))
        self.assertTrue(torch.isfinite(result.watch_to_phone_loss))
        self.assertGreaterEqual(result.loss.item(), 0.0)

    def test_symmetric_info_nce_requires_batch_negatives(self) -> None:
        z_phone = torch.randn(1, 64)
        z_watch = torch.randn(1, 64)
        with self.assertRaises(ValueError):
            symmetric_info_nce_loss(z_phone, z_watch)


if __name__ == "__main__":
    unittest.main()
