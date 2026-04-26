import h5py
import numpy as np
import pickle


hdf5_file_path = "/home/jau/dyros/src/fr3_teleop/data/hdf5/mobile_base_20260421_183129_0/episode_0.hdf5"
dataset_stats_path = "/home/jau/dyros/src/act/ckpt/mobile_base_20260421_183129_run5/dataset_stats.pkl"


num = 50


stats = pickle.load(open(dataset_stats_path, "rb"))
print("action_mean:", stats["action_mean"])
print("action_std :", stats["action_std"])
print()

with h5py.File(hdf5_file_path, "r") as f:
    print("Keys:", list(f.keys()))
    print("Action type:", type(f["action"]))

    actions = f["action"][:]   # (T, 7)
    print("Action shape:", actions.shape)

    print(f"\nFirst {num} action vectors:")
    for i in range(min(num, actions.shape[0])):
        print(f"{i:02d}: {np.round(actions[i], 4)}")

    print(f"\nFirst {num} z-values:")
    print(np.round(actions[:num, 2], 4))

    print("\nStats (first 50 timesteps):")
    print("mean:", np.round(actions[:50].mean(axis=0), 4))
    print("std :", np.round(actions[:50].std(axis=0), 4))