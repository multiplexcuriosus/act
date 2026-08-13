# Sparse ball input

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
