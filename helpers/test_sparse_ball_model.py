"""Sparse ACT model fixture; skipped when the training stack is unavailable."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from policy import ACTPolicy  # noqa: E402


@pytest.mark.parametrize('history_length,state_dim', [(3, 21), (32, 91)])
def test_sparse_act_forward_has_no_visual_backbone(history_length, state_dim):
    config = {
        "lr": 1e-4, "num_queries": 30, "kl_weight": 1,
        "hidden_dim": 32, "dim_feedforward": 64, "lr_backbone": 1e-5,
        "backbone": "resnet18", "enc_layers": 1, "dec_layers": 1,
        "nheads": 8, "camera_names": ["sparse_ball"],
        "input_modality": "sparse_ball", "state_dim": state_dim, "action_dim": 1,
        "use_bce_last_action_dim": False, "device": "cpu",
        "image_channels": 0, "sparse_feature_dim": 4,
        "sparse_history_length": history_length, "sparse_mean": [0, 0, 0, 0],
        "sparse_std": [1, 1, 1, 1],
    }
    policy = ACTPolicy(config)
    assert policy.model.backbones is None
    output = policy(torch.zeros(2, state_dim), torch.zeros(2, history_length, 4))
    assert output.shape == (2, 30, 1)
