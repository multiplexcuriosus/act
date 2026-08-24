#!/usr/bin/env python3
"""Minimal one-batch sparse ACT parameter/CUDA-memory profiler."""

import argparse
import json

import torch

from policy import ACTPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-rate-hz', type=int, choices=(30, 60), default=60)
    parser.add_argument('--sparse-history-capacity', type=int, default=32)
    parser.add_argument('--state-dim', type=int, default=91)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'lr': 1e-5, 'num_queries': args.policy_rate_hz, 'kl_weight': 10,
        'hidden_dim': 512, 'dim_feedforward': 3200, 'lr_backbone': 1e-5,
        'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7,
        'nheads': 8, 'camera_names': ['sparse_ball'],
        'input_modality': 'sparse_ball', 'state_dim': args.state_dim,
        'action_dim': 1, 'use_bce_last_action_dim': False,
        'device': device.type, 'image_channels': 0, 'sparse_feature_dim': 4,
        'sparse_history_length': args.sparse_history_capacity,
        'sparse_mean': [0] * 4, 'sparse_std': [1] * 4,
    }
    policy = ACTPolicy(config).to(device).eval()
    qpos = torch.zeros(args.batch_size, args.state_dim, device=device)
    sparse = torch.zeros(args.batch_size, args.sparse_history_capacity, 4, device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = policy(qpos, sparse)
    report = {
        'parameter_count': sum(p.numel() for p in policy.parameters()),
        'device': device.type,
        'cuda_allocated_bytes': (torch.cuda.memory_allocated(device) if device.type == 'cuda' else None),
        'cuda_peak_bytes': (torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None),
        'input_shapes': {'qpos': list(qpos.shape), 'sparse': list(sparse.shape)},
        'output_shape': list(output.shape),
        'history': {'mode': 'm_window', 'history_ms': 200,
                    'sparse_capacity': args.sparse_history_capacity,
                    'policy_rate_hz': args.policy_rate_hz},
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
