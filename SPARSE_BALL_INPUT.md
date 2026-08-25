# Sparse ball input

## Live rollout provenance

When latency tracing is enabled, the sparse rollout writes the exact causal
event observation selected at each policy tick into `LatencyTrace.detail_json`.
`event_source_timestamp_ns` comes from the subscribed `PointStamped.header.stamp`,
and `event_age_sec` is policy-tick ROS time minus that source time.  The local
`event_observation_sequence` advances when the selected source timestamp changes;
it is **not** an OpenMV hardware packet ID.

The current `/openmv_cam/event_tracker/ball_2d_px` type is
`geometry_msgs/msg/PointStamped` and therefore exposes no tracker update ID or
hardware packet ID.  To expose one, the upstream OpenMV event-tracker publisher
must publish a custom stamped message containing the point plus a `uint64`
tracker update/hardware packet ID (or publish an equivalently header-synchronized
companion metadata topic).  The rollout can then copy that field into
`event_source_update_id`; changing detector-latency reporting alone is not a
freshness signal.

`helpers/export_act_event_provenance.py` converts decoded ACT `LatencyTrace`
JSON/JSONL records to one CSV row per trace for causal probability analysis.

DLAB trains ACT directly from sparse RGB or event tracker observations while
keeping RGB images and optional dense event tensors in the same HDF5 episode.
The integrated bag converter is the authoritative writer; sparse enrichment is
additive and does not replace `/observations/images/rgb`.

Raw tracking data is stored below `/observations/sparse_tracking` as pixel
coordinates, a `uint8` validity mask, and source timestamps in seconds:

- `<source>_2d_px`: `[u_px, v_px]`
- `<source>_valid`: `0` or `1`
- `<source>_source_timestamps`: PointStamped source time (seconds)

The training loader converts those raw values to a three-sample history whose
four features are `[u_norm, v_norm, valid, observation_age_sec]`. Invalid or
stale samples are `[0, 0, 0, max_observation_age_sec]`. Coordinates use pixel
centres normalized independently to `[-1, 1]`, with `u` increasing rightward
and `v` increasing downward.

Use `grid_train_sparse_ball_intercept.sh` for sparse training. It selects
`--camera_names sparse_ball --input_modality sparse_ball` and requires an
explicit `--sparse_source rgb|event`; it does not train from dense event images.

`helpers/add_sparse_ball_to_intercept_hdf5.py` from the historical sparse
branch is intentionally not part of the canonical workflow. Existing datasets
should be regenerated or migrated with an explicitly reviewed compatibility
tool rather than maintaining a second writer with a divergent schema.
