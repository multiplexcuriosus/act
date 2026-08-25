# Sparse ball input

## Live rollout provenance

Event sparse rollout defaults to `/openmv_cam/event_tracker/update`, whose exact
type is `openmv_cam/msg/EventTrackerUpdate` (Python import
`from openmv_cam.msg import EventTrackerUpdate`). Every callback, including an
invalid result, is retained in the tracker-update provenance buffer. The latest
tracker update therefore describes what ACT has received, while the
policy-selected observation describes the latest valid result causally available
at the policy/history target time. An invalid latest update does not erase the
previous valid observation: the existing max-age hold semantics continue to feed
that valid observation until it becomes stale. This keeps ACT prediction behavior
unchanged.

`availability_timestamp_ns` (equivalent to `header.stamp`) is the tracker result's
host ROS availability/source timestamp. `*_receipt_timestamp_ns` is sampled by
the rollout callback and measures when ACT received it. Their difference exposes
ROS callback/transport delay; policy time minus availability time exposes source
age. The sensor-window microsecond fields are in the GenX320 sensor clock domain
and must not be subtracted from ROS time.

`tracker_update_id` is a monotonic tracker-process update sequence.
`source_packet_id`, when `source_packet_id_valid` is true, is the EVR1 hardware/raw
packet sequence. The compatibility `event_observation_sequence` is only a local
rollout sequence and is **not** a hardware packet ID. Packet IDs remain unavailable
for processed EVT1 and current HDF5 sources, as stated by the upstream message.

For old deployments, `--legacy_event_pointstamped` subscribes to the unchanged
`--sparse_topic` (normally `/openmv_cam/event_tracker/ball_2d_px`). This fallback
cannot distinguish invalid tracker updates and has no tracker or hardware packet
ID; it labels that limitation explicitly in the rejection-reason provenance.

`helpers/export_act_event_provenance.py` converts decoded ACT `LatencyTrace`
JSON/JSONL records to one CSV row per trace. Add `--summary-json summary.json` to
compute the requested conditional probabilities, held/invalid fractions, hold
duration distribution, and source-age/transport-delay percentiles.

The provenance separates four bottlenecks:

1. Tracker generation: inspect latest update ID, validity, rejection reason,
   event count, and availability/source age.
2. ROS delivery/selection: inspect callback delay and compare latest update with
   the causally selected/held observation and its full three-slot history.
3. ACT sensitivity: compare `sparse_input_changed` with
   `prediction_changed_gt_0_1mm`.
4. Downstream execution: join prediction time/value to the independently traced
   `track_s` stream. That final association remains timestamp-based because the
   controller interface is intentionally unchanged.

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
