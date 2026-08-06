#!/usr/bin/env python3

import unittest

import torch

from policy import ACTPolicy


class PolicyPreprocessingTests(unittest.TestCase):
    def test_sparse_ball_uses_sparse_statistics(self):
        policy = object.__new__(ACTPolicy)
        policy.input_modality = "sparse_ball"
        policy.sparse_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
        policy.sparse_std = torch.tensor([2.0, 4.0, 5.0, 8.0])
        policy.normalize = None
        image = torch.tensor([[[3.0, 6.0, 8.0, 12.0]]])

        actual = policy.preprocess_image(image)

        torch.testing.assert_close(
            actual,
            torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]),
        )

    def test_dense_input_uses_configured_normalizer(self):
        policy = object.__new__(ACTPolicy)
        policy.input_modality = "event"
        policy.normalize = lambda image: image + 7.0
        image = torch.tensor([1.0])

        torch.testing.assert_close(
            policy.preprocess_image(image),
            torch.tensor([8.0]),
        )


if __name__ == "__main__":
    unittest.main()
