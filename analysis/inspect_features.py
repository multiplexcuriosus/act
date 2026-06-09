import numpy as np

d = np.load("/home/jau/dyros/data/features/event_cnn_features.npz")
X = d["cnn_proj_pooled"]

print(X.shape)
print(np.linalg.norm(X, axis=1).mean())
print(np.linalg.norm(X, axis=1).std())