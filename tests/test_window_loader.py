from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.data import ContrastiveWindowDataset, SupervisedWindowDataset, WindowRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WindowRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = WindowRepository(PROJECT_ROOT / "artifacts" / "windows")

    def test_manifest_counts_match_loaded_splits(self) -> None:
        counts = self.repository.validate_against_manifest()
        self.assertEqual(counts, self.repository.manifest["split_window_counts"])

    def test_sample_contract_for_each_split(self) -> None:
        for split in ("train", "val", "test"):
            split_data = self.repository.load_split(split)
            self.assertGreater(len(split_data), 0)
            sample = split_data.samples[0]
            self.assertEqual(sample.x_fusion.shape, (12, 60))
            self.assertEqual(sample.x_phone.shape, (6, 60))
            self.assertEqual(sample.x_watch.shape, (6, 60))
            np.testing.assert_allclose(sample.x_fusion[:6], sample.x_phone)
            np.testing.assert_allclose(sample.x_fusion[6:], sample.x_watch)
            self.assertEqual(sample.split, split)

    def test_dataset_views_match_loader_contract(self) -> None:
        split_data = self.repository.load_split("train")
        supervised_phone = SupervisedWindowDataset(split_data, channel_mode="phone")[0]
        supervised_watch = SupervisedWindowDataset(split_data, channel_mode="watch")[0]
        supervised_fusion = SupervisedWindowDataset(split_data, channel_mode="fusion")[0]
        contrastive = ContrastiveWindowDataset(split_data)[0]

        self.assertEqual(supervised_phone.x.shape, (6, 60))
        self.assertEqual(supervised_watch.x.shape, (6, 60))
        self.assertEqual(supervised_fusion.x.shape, (12, 60))
        np.testing.assert_allclose(supervised_phone.x, contrastive.x_phone)
        np.testing.assert_allclose(supervised_watch.x, contrastive.x_watch)


if __name__ == "__main__":
    unittest.main()
