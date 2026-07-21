import glob
import h5py
import numpy as np

paths = glob.glob("/home/dyros/Data/jg_data/hdf5/recording_20260716_161544_bag/episode_0.hdf5")

flags = []
for path in paths:
    with h5py.File(path, "r") as f:
        action = f["action"][:]

        # New combined schema
        if action.ndim == 2 and action.shape[1] == 2:
            episode_flags = action[:, 1]
        # Old split schema
        else:
            episode_flags = f["action_is_commanded"][:].reshape(-1)

        flags.append(episode_flags)

flags = np.concatenate(flags)

print("unique:", np.unique(flags))
print("negative:", np.sum(flags == 0))
print("positive:", np.sum(flags == 1))
print("positive fraction:", np.mean(flags))