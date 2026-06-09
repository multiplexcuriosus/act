import h5py

f = h5py.File("/home/dyros/Data/jg_data/hdf5/enclosure_tennisball_red_ring_smooth/episode_0.hdf5", "r")
print(list(f.keys()))
print(list(f["observations"].keys()))
print(list(f["observations"]["images"].keys()))
print(list(f["action"].keys()))
print(f["observations"]["qpos"].shape)
print(f["observations"]["images"]["rgb"].shape)
print(f["observations"]["images"]["event"].shape)
print(f["action"]["combined"].shape)
print(f.attrs["joint_names"])
f.close()