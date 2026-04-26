import h5py

f = h5py.File("/home/jau/dyros/src/fr3_teleop/data/hdf5/mobile_base_20260421_183129_0/episode_0.hdf5", "r")
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