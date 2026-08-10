# Sparse ball input for interception ACT

Select the mode with `--camera_names sparse_ball` for training and
`--input_modality sparse_ball` for rollout. The canonical tensor is `[B,3,6]` at
grid offsets `[-6,-3,0]`, ordered as `u, v, du_dt, dv_dt, valid,
observation_age`.

Pixel coordinates use `2*p/(size-1)-1`. Velocity is the difference between the
newest two causal normalized positions divided by their header-timestamp
difference (normalized-image units/s). At policy time `t`, only observations
whose header timestamp is `<=t` are visible. Position is held from the newest
causal observation, velocity is zero without two observations, age is clipped
to the configured maximum, and `valid=0` when no observation exists or it is
older than that maximum. Before the first observation the representation is
`[0,0,0,0,0,max_age]`. ROS receipt time is never used for ball association or
velocity.

First augment converted interception episodes:

```bash
python3 helpers/add_sparse_ball_to_intercept_hdf5.py --bag BAG --hdf5_dir DATA \
  --image_width 640 --image_height 480
```

Train with `./train_sparse_ball_intercept.sh DATA CKPTS`. Roll out with the
existing node plus `--input_modality sparse_ball --ckpt_dir CKPTS`. The rollout
subscribes to `/ball_tracker2/ball_2d_px` by default and does not subscribe to an
RGB image.

Sparse mean/std are computed from the training split and saved in
`dataset_stats.pkl`, together with feature, timestamp, missing-data, dimensions,
topic, and convention metadata. Sparse and dense checkpoints are intentionally
incompatible and rejected by contract validation.

The output remains 30 relative `delta-s` tokens with action dimension one.
Rollout re-anchors them to live TCP `s`, then applies the unchanged lookahead,
temporal aggregation, prediction topics, and controller-facing message types.
