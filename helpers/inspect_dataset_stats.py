import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to dataset_stats.pkl")
args = parser.parse_args()

with open(args.path, "rb") as f:
    s = pickle.load(f)

print("Dataset_stats.pkl")
print("===================")
for k in ["action_mean", "action_std", "qpos_mean", "qpos_std"]:
    v = np.array(s[k])
    print(k, v)

print("\nRepresentation contract")
print("=======================")
for k in [
    "input_modality",
    "event_representation",
    "event_horizon_ms",
    "event_temporal_bins",
    "event_bin_width_ms",
    "event_spatial_height",
    "event_spatial_width",
    "event_channel_order",
    "event_polarity_encoding",
    "event_scaling",
    "event_clip_count",
    "event_neutral_u8",
    "event_sampling_policy",
    "visual_history_frames",
    "visual_history_offsets",
    "qpos_history_frames",
    "qpos_history_offsets",
    "channels_per_visual_frame",
    "image_channels",
    "image_normalization",
]:
    if k in s:
        print(f"{k}: {s[k]}")
