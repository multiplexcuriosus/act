# Datset_stats.pkl
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