"""Sparse ACT model fixture; skipped when the training stack is unavailable."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from policy import ACTPolicy  # noqa: E402


def test_sparse_act_forward_has_no_visual_backbone():
    config = {
        "lr": 1e-4, "num_queries": 30, "kl_weight": 1,
        "hidden_dim": 32, "dim_feedforward": 64, "lr_backbone": 1e-5,
        "backbone": "resnet18", "enc_layers": 1, "dec_layers": 1,
        "nheads": 8, "camera_names": ["sparse_ball"],
        "input_modality": "sparse_ball", "state_dim": 21, "action_dim": 1,
        "use_bce_last_action_dim": False, "device": "cpu",
        "image_channels": 0, "sparse_feature_dim": 4,
        "sparse_history_length": 3, "sparse_mean": [0, 0, 0, 0],
        "sparse_std": [1, 1, 1, 1],
    }
    policy = ACTPolicy(config)
    assert policy.model.backbones is None
    output = policy(torch.zeros(2, 21), torch.zeros(2, 3, 4))
    assert output.shape == (2, 30, 1)


def test_sparse_act_forward_supports_dynamic_m_window_shape():
    config = {
        "lr": 1e-4, "num_queries": 60, "kl_weight": 1,
        "hidden_dim": 32, "dim_feedforward": 64, "lr_backbone": 1e-5,
        "backbone": "resnet18", "enc_layers": 1, "dec_layers": 1,
        "nheads": 8, "camera_names": ["sparse_ball"],
        "input_modality": "sparse_ball", "state_dim": 49, "action_dim": 1,
        "use_bce_last_action_dim": False, "device": "cpu",
        "image_channels": 0, "sparse_feature_dim": 4,
        "sparse_history_length": 32, "sparse_mean": [0, 0, 0, 0],
        "sparse_std": [1, 1, 1, 1],
    }
    policy = ACTPolicy(config)
    output = policy(torch.zeros(2, 49), torch.zeros(2, 32, 4))
    assert output.shape == (2, 60, 1)
