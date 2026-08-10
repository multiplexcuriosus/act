#!/usr/bin/env python3
"""Add causal sparse-ball arrays to existing interception HDF5 episodes from a ROS bag."""
import argparse
from pathlib import Path
import sys
import h5py
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparse_ball import BallObservation, sparse_feature_at_time, sparse_metadata


def read_ball_observations(bag, storage_id, topic):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage_id),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic not in types:
        raise RuntimeError(f"ball topic is missing from bag: {topic}")
    if types[topic] != 'geometry_msgs/msg/PointStamped':
        raise RuntimeError(f"{topic} must be geometry_msgs/msg/PointStamped, got {types[topic]}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    cls = get_message(types[topic])
    result = []
    while reader.has_next():
        _topic, raw, _bag_ns = reader.read_next()
        msg = deserialize_message(raw, cls)
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        result.append(BallObservation(stamp, float(msg.point.x), float(msg.point.y)))
    return result


def augment_episode(path, observations, args):
    with h5py.File(path, 'r+') as root:
        grid = np.asarray(root['/observations/timestamps'][()], dtype=np.float64)
        features = np.stack([sparse_feature_at_time(observations, t, args.image_width,
                                                    args.image_height, args.max_observation_age_sec)
                             for t in grid]).astype(np.float32)
        key = '/observations/sparse_ball'
        if key in root:
            if not args.overwrite:
                raise RuntimeError(f"{path}: {key} exists; pass --overwrite")
            del root[key]
        root.create_dataset(key, data=features, dtype=np.float32)
        metadata = sparse_metadata(args.image_width, args.image_height,
                                   args.max_observation_age_sec, args.ball_topic)
        for name, value in metadata.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                value = np.asarray(value, dtype=h5py.string_dtype('utf-8'))
            root.attrs[name] = value
        root.create_dataset('/observations/sparse_ball_source_timestamps',
                            data=grid - features[:, 5], dtype=np.float64)
        root.create_dataset('/observations/sparse_ball_valid', data=features[:, 4], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--hdf5_dir', required=True)
    parser.add_argument('--storage_id', choices=['mcap', 'sqlite3'], default='mcap')
    parser.add_argument('--ball_topic', default='/ball_tracker2/ball_2d_px')
    parser.add_argument('--image_width', type=int, required=True)
    parser.add_argument('--image_height', type=int, required=True)
    parser.add_argument('--max_observation_age_sec', type=float, default=.2)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    observations = read_ball_observations(args.bag, args.storage_id, args.ball_topic)
    if not observations:
        raise RuntimeError('no ball observations found')
    paths = sorted(Path(args.hdf5_dir).glob('episode_*.hdf5'))
    if not paths:
        raise RuntimeError('no episode_*.hdf5 files found')
    for path in paths:
        augment_episode(path, observations, args)
        print(path)


if __name__ == '__main__':
    main()
