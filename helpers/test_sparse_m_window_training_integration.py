"""End-to-end synthetic coverage for sparse M-window ACT training."""

import h5py
import numpy as np
import pytest
import sys
import types
from pathlib import Path

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

ACT_DIR = Path(__file__).resolve().parents[1]
DETR_DIR = ACT_DIR / "detr"
for path in (ACT_DIR, DETR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The training modules import IPython only to expose a developer breakpoint.
# Keep that optional debugging dependency out of this CPU integration test.
if "IPython" not in sys.modules:
    try:
        import IPython  # noqa: F401
    except ImportError:
        sys.modules["IPython"] = types.SimpleNamespace(embed=lambda: None)
try:
    import roma.mappings  # noqa: F401
except ImportError:
    roma_module = types.ModuleType("roma")
    mappings_module = types.ModuleType("roma.mappings")
    mappings_module.special_gramschmidt = lambda value: value
    roma_module.mappings = mappings_module
    sys.modules["roma"] = roma_module
    sys.modules["roma.mappings"] = mappings_module

from policy import ACTPolicy  # noqa: E402
from sparse_ball import qpos_history_offsets_for_window  # noqa: E402
from utils import (  # noqa: E402
    EpisodicInterceptDataset,
    _read_raw_sparse_episode,
    get_intercept_norm_stats,
)


def _write_m_window_episode(path, rate):
    steps = rate + 2
    grid = 10.0 + np.arange(steps, dtype=np.float64) / rate
    qpos = np.arange(steps * 7, dtype=np.float32).reshape(steps, 7) / 100.0
    action = np.linspace(0.1, 0.2, steps, dtype=np.float32)[:, None]

    # Irregular literal observations include invalid points and observations
    # beyond early policy anchors. The window constructor must remain causal.
    raw_timestamps = np.asarray(
        [9.90, 9.999, 10.004, 10.011, 10.027, 10.052,
         10.101, 10.199, 10.205, 10.401, 11.001],
        dtype=np.float64,
    )
    raw_points = np.stack(
        [np.linspace(10, 110, len(raw_timestamps)),
         np.linspace(20, 120, len(raw_timestamps))],
        axis=1,
    ).astype(np.float32)
    raw_valid = np.asarray([1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.uint8)

    with h5py.File(path, "w") as root:
        root.attrs.update(
            sim=False,
            action_type="measured_tcp_s_absolute",
            action_representation="absolute",
            action_positive_direction="robot_base_positive_x",
            policy_rate_hz=rate,
        )
        root.create_dataset("/action", data=action)
        root.create_dataset("/observations/qpos", data=qpos)
        root.create_dataset("/observations/timestamps", data=grid)
        sparse = root.require_group("/observations/sparse_tracking")
        sparse.attrs["rgb_width_px"] = 320
        sparse.attrs["rgb_height_px"] = 240
        sparse.create_dataset("raw_rgb_timestamps", data=raw_timestamps)
        sparse.create_dataset("raw_rgb_2d_px", data=raw_points)
        sparse.create_dataset("raw_rgb_valid", data=raw_valid)


def _policy_config(rate, state_dim, stats):
    return {
        "lr": 1e-4,
        "num_queries": rate,
        "kl_weight": 1,
        "hidden_dim": 32,
        "dim_feedforward": 64,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 1,
        "dec_layers": 1,
        "nheads": 8,
        "camera_names": ["sparse_ball"],
        "input_modality": "sparse_ball",
        "state_dim": state_dim,
        "action_dim": 1,
        "use_bce_last_action_dim": False,
        "device": "cpu",
        "image_channels": 0,
        "sparse_feature_dim": 4,
        "sparse_history_length": 32,
        "sparse_mean": stats["sparse_mean"],
        "sparse_std": stats["sparse_std"],
    }


@pytest.mark.parametrize("rate,state_dim", [(30, 49), (60, 91)])
def test_raw_hdf5_to_sparse_act_training_step(tmp_path, rate, state_dim):
    episode = tmp_path / f"episode_{rate}.hdf5"
    _write_m_window_episode(episode, rate)
    offsets = qpos_history_offsets_for_window(rate, 200)

    with h5py.File(episode, "r") as root:
        raw = _read_raw_sparse_episode(root, str(episode), "rgb")
    assert len(raw) == 11
    assert all(a.source_timestamp <= b.source_timestamp for a, b in zip(raw, raw[1:]))

    stats = get_intercept_norm_stats(
        [str(episode)],
        chunk_size=rate,
        history_offsets=offsets,
        visual_history_offsets=(0,),
        input_modality="sparse_ball",
        sparse_source="rgb",
        sparse_history_mode="m_window",
        history_ms=200,
        sparse_history_capacity=32,
        policy_rate_hz=rate,
    )
    assert stats["qpos_mean"].shape == (state_dim,)
    assert stats["sparse_mean"].shape == (4,)
    assert stats["state_dim"] == state_dim
    assert stats["sparse_history_length"] == 32

    dataset = EpisodicInterceptDataset(
        [str(episode)], ["sparse_ball"], rate, stats,
        history_offsets=offsets,
        visual_history_offsets=(0,),
        input_modality="sparse_ball",
        sparse_source="rgb",
        sparse_history_mode="m_window",
        history_ms=200,
        sparse_history_capacity=32,
    )
    sparse, qpos, action, is_pad = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    assert sparse.shape == (1, 32, 4)
    assert qpos.shape == (1, state_dim)
    assert action.shape == (1, rate, 1)
    assert is_pad.shape == (1, rate)

    policy = ACTPolicy(_policy_config(rate, state_dim, stats))
    assert policy(qpos, sparse).shape == (1, rate, 1)
    optimizer = policy.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss = policy(qpos, sparse, action, is_pad)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    optimizer.step()
