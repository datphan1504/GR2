"""Unit tests for model architecture construction and mock inference."""

import unittest
import numpy as np

from src.models.architectures import build_var_cnn, build_df_model, build_model


class TestInference(unittest.TestCase):

    def setUp(self):
        self.config = {
            "num_mon_sites": 10,
            "num_unmon_sites_test": 0,
            "num_unmon_sites_train": 0,
            "seq_length": 5000,
            "mixture": [["dir", "metadata"]],
            "dir_dilations": False,
            "time_dilations": False,
            "learning_rate": 0.001,
            "dense_units": 64,
            "dropout_rate": 0.2
        }

    def test_build_var_cnn(self):
        model = build_var_cnn(self.config, mixture_num=0)
        self.assertIsNotNone(model)

        # Mock inputs
        batch_size = 2
        mock_dir = np.random.choice([-1, 0, 1], size=(batch_size, 5000, 1)).astype(np.float32)
        mock_meta = np.random.randn(batch_size, 7).astype(np.float32)

        preds = model.predict({'dir_input': mock_dir, 'metadata_input': mock_meta})
        self.assertEqual(preds.shape, (batch_size, 10))
        # Probabilities should sum to approximately 1
        np.testing.assert_allclose(np.sum(preds, axis=1), np.ones(batch_size), atol=1e-5)

    def test_build_df_model(self):
        model = build_df_model(self.config)
        self.assertIsNotNone(model)

        batch_size = 2
        mock_dir = np.random.choice([-1, 0, 1], size=(batch_size, 5000, 1)).astype(np.float32)

        preds = model.predict(mock_dir)
        self.assertEqual(preds.shape, (batch_size, 10))
        np.testing.assert_allclose(np.sum(preds, axis=1), np.ones(batch_size), atol=1e-5)

    def test_factory_build_model(self):
        self.config['model_name'] = 'var-cnn'
        model1 = build_model(self.config)
        self.assertIsNotNone(model1)

        self.config['model_name'] = 'df'
        model2 = build_model(self.config)
        self.assertIsNotNone(model2)


if __name__ == "__main__":
    unittest.main()
