import numpy as np
import torch
import os
import glob
import re
import h5py
import cv2
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from sparse_ball import SPARSE_BALL_FEATURE_DIM, SPARSE_BALL_FEATURE_NAMES

import IPython
e = IPython.embed

color_transform = transforms.ColorJitter(brightness=0.5,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1,
                          )

DEFAULT_IMAGE_SIZE = (320, 320)


def canonical_image_size(image_size=None):
    """Normalize image_size to a (H, W) tuple."""
    if image_size is None:
        return DEFAULT_IMAGE_SIZE
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        return (image_size, image_size)
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        h, w = int(image_size[0]), int(image_size[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size dimensions must be positive, got ({h}, {w})")
        return (h, w)
    raise ValueError(f"Unsupported image_size type/shape: {image_size!r}")



DEFAULT_JOINT_DATA_CONFIG = {
    "qpos_dim": None,
    "action_dim": None,
    "qpos_indices": None,
    "action_indices": None,
    "action_key": "/action",
    "command_flag_key": None,
}


INTERCEPT_HISTORY_OFFSETS_DEFAULT = (-6, -3, 0)
INTERCEPT_EVENT_WINDOWS_MS_DEFAULT = [50.0, 100.0, 200.0]
INTERCEPT_EVENT_REQUIRED_ATTRS = {
    'event_representation': 'shifted_3chef_signed',
    'event_frame_mode': 'shifted',
    'event_channel_order': 'recent_to_oldest',
    'event_scaling': 'signed_log1p_fixed_clip',
    'event_neutral_u8': 128,
    'event_sampling_policy': 'latest_packet_at_or_before_grid_time',
}
INTERCEPT_REQUIRED_ROOT_METADATA = {
    'action_type': 'measured_tcp_s_absolute',
    'action_representation': 'absolute',
    'action_positive_direction': 'robot_base_positive_x',
}


def _decode_h5_attr(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def _read_h5_root_attr(root, key):
    if key not in root.attrs:
        return None
    return _decode_h5_attr(root.attrs[key])


def _validate_intercept_root_metadata(root, dataset_path):
    missing_keys = [
        key for key in INTERCEPT_REQUIRED_ROOT_METADATA
        if key not in root.attrs
    ]
    if missing_keys:
        raise ValueError(
            f"Missing interception metadata at HDF5 root in {dataset_path}: {missing_keys}. "
            "This dataset requires episodes converted with the new interception converter that sets root attrs "
            "action_type, action_representation, and action_positive_direction."
        )

    action_type = str(_read_h5_root_attr(root, 'action_type'))
    action_representation = str(_read_h5_root_attr(root, 'action_representation'))
    action_positive_direction = str(_read_h5_root_attr(root, 'action_positive_direction'))

    if action_positive_direction == 'table_frame_positive_s':
        raise ValueError(
            f"Rejected {dataset_path}: action_positive_direction=table_frame_positive_s is unsupported for interception training. "
            "Expected robot_base_positive_x from the converter metadata."
        )

    if action_type != INTERCEPT_REQUIRED_ROOT_METADATA['action_type']:
        raise ValueError(
            f"Rejected {dataset_path}: action_type={action_type!r}. "
            f"Expected {INTERCEPT_REQUIRED_ROOT_METADATA['action_type']!r}."
        )
    if action_representation != INTERCEPT_REQUIRED_ROOT_METADATA['action_representation']:
        raise ValueError(
            f"Rejected {dataset_path}: action_representation={action_representation!r}. "
            f"Expected {INTERCEPT_REQUIRED_ROOT_METADATA['action_representation']!r}."
        )
    if action_positive_direction != INTERCEPT_REQUIRED_ROOT_METADATA['action_positive_direction']:
        raise ValueError(
            f"Rejected {dataset_path}: action_positive_direction={action_positive_direction!r}. "
            f"Expected {INTERCEPT_REQUIRED_ROOT_METADATA['action_positive_direction']!r}."
        )


def compute_history_indices(anchor_t, history_offsets):
    return [max(0, int(anchor_t) + int(offset)) for offset in history_offsets]


def _build_joint_data_config(
    qpos_dim=None,
    action_dim=None,
    qpos_indices=None,
    action_indices=None,
    action_key=None,
    command_flag_key=None,
):
    cfg = dict(DEFAULT_JOINT_DATA_CONFIG)
    cfg["qpos_dim"] = qpos_dim
    cfg["action_dim"] = action_dim
    cfg["action_key"] = action_key if action_key is not None else cfg["action_key"]
    cfg["command_flag_key"] = command_flag_key

    # qpos/action indexing is intentionally decoupled.
    if qpos_indices is not None:
        cfg["qpos_indices"] = list(qpos_indices)
    elif qpos_dim is not None:
        cfg["qpos_indices"] = list(range(qpos_dim))

    if action_indices is not None:
        cfg["action_indices"] = list(action_indices)
    elif action_dim is not None:
        cfg["action_indices"] = list(range(action_dim))

    return cfg


def _read_action_array(root, action_key='/action', command_flag_key=None):
    if action_key not in root:
        raise ValueError(f"Missing required dataset: {action_key}")

    action = np.asarray(root[action_key][()], dtype=np.float32)
    if action.ndim == 0:
        raise ValueError(f"Action dataset {action_key} must have a time dimension, got shape {action.shape}")
    action = action.reshape(action.shape[0], -1)

    if command_flag_key is not None:
        if command_flag_key not in root:
            raise ValueError(f"Missing required dataset: {command_flag_key}")

        flag = np.asarray(root[command_flag_key][()], dtype=np.float32)
        if flag.ndim == 0:
            raise ValueError(
                f"Command-flag dataset {command_flag_key} must have a time dimension, got shape {flag.shape}"
            )
        flag = flag.reshape(flag.shape[0], -1)
        if flag.shape[1] != 1:
            raise ValueError(
                f"Command-flag dataset {command_flag_key} must have width 1 after reshape, got shape {flag.shape}"
            )
        if flag.shape[0] != action.shape[0]:
            raise ValueError(
                f"Unequal action/flag lengths for {action_key} and {command_flag_key}: "
                f"{action.shape[0]} != {flag.shape[0]}"
            )

        action = np.concatenate([action, flag], axis=1)

    return action


def _validate_joint_episode_structure(
    root,
    dataset_path,
    camera_names,
    qpos_dim=None,
    action_dim=None,
    action_key='/action',
    command_flag_key=None,
):
    if '/observations/qpos' not in root:
        raise ValueError(f"Missing required dataset: /observations/qpos in {dataset_path}")

    qpos = np.asarray(root['/observations/qpos'][()], dtype=np.float32)
    if qpos.ndim != 2:
        raise ValueError(f"Expected /observations/qpos to have shape (T, Q) in {dataset_path}, got {qpos.shape}")

    action = _read_action_array(root, action_key=action_key, command_flag_key=command_flag_key)
    if action.ndim != 2:
        raise ValueError(f"Expected dense action array to have shape (T, A) in {dataset_path}, got {action.shape}")

    if command_flag_key is not None and action.shape[1] != 2:
        raise ValueError(
            f"Intercept actions must have width 2 after concatenation in {dataset_path}, got {action.shape[1]}"
        )

    if qpos.shape[0] != action.shape[0]:
        raise ValueError(
            f"Unequal numbers of qpos and dense actions in {dataset_path}: {qpos.shape[0]} != {action.shape[0]}"
        )

    for cam_name in camera_names:
        if cam_name.endswith('stereo'):
            image_keys = [
                f'/observations/images/{cam_name[:-6]}left',
                f'/observations/images/{cam_name[:-6]}right',
            ]
        else:
            image_keys = [f'/observations/images/{cam_name}']

        for image_key in image_keys:
            if image_key not in root:
                raise ValueError(f"Missing required dataset: {image_key} in {dataset_path}")
            image_steps = root[image_key].shape[0]
            if image_steps != action.shape[0]:
                raise ValueError(
                    f"Unequal numbers of images and dense actions in {dataset_path} for {image_key}: "
                    f"{image_steps} != {action.shape[0]}"
                )

    if qpos_dim is not None and qpos.shape[1] != qpos_dim:
        raise ValueError(
            f"state_dim mismatch for {dataset_path}: expected {qpos_dim}, found qpos width {qpos.shape[1]}"
        )

    if action_dim is not None and action.shape[1] != action_dim:
        raise ValueError(
            f"action_dim mismatch for {dataset_path}: expected {action_dim}, found action width {action.shape[1]}"
        )

    return qpos, action


def _extract_intercept_event_metadata(root, dataset_path):
    metadata = {}
    for key, expected in INTERCEPT_EVENT_REQUIRED_ATTRS.items():
        if key not in root.attrs:
            raise ValueError(
                f"Missing interception event metadata at HDF5 root in {dataset_path}: {key}"
            )
        value = _decode_h5_attr(root.attrs[key])
        if value != expected:
            raise ValueError(
                f"Interception event metadata mismatch in {dataset_path} for {key}: "
                f"expected {expected!r}, found {value!r}"
            )
        metadata[key] = value

    if 'event_frame_windows_ms' not in root.attrs:
        raise ValueError(
            f"Missing interception event metadata at HDF5 root in {dataset_path}: event_frame_windows_ms"
        )
    windows = np.asarray(root.attrs['event_frame_windows_ms'], dtype=np.float64).reshape(-1)
    if windows.shape != (3,):
        raise ValueError(
            f"Interception event metadata event_frame_windows_ms must have shape (3,), got {windows.shape} in {dataset_path}"
        )
    expected_windows = np.asarray(INTERCEPT_EVENT_WINDOWS_MS_DEFAULT, dtype=np.float64)
    if not np.allclose(windows, expected_windows, atol=1e-9):
        raise ValueError(
            f"Interception event metadata mismatch in {dataset_path} for event_frame_windows_ms: "
            f"expected {expected_windows.tolist()}, found {windows.tolist()}"
        )
    metadata['event_frame_windows_ms'] = windows.astype(np.float32).tolist()

    if 'event_clip_count' not in root.attrs:
        raise ValueError(
            f"Missing interception event metadata at HDF5 root in {dataset_path}: event_clip_count"
        )
    event_clip_count = float(root.attrs['event_clip_count'])
    if not np.isfinite(event_clip_count) or event_clip_count <= 0.0:
        raise ValueError(
            f"Invalid event_clip_count in {dataset_path}: expected positive finite value, got {event_clip_count}"
        )
    metadata['event_clip_count'] = event_clip_count

    return metadata


def _validate_intercept_episode_structure(
    root,
    dataset_path,
    modality='rgb',
    expected_event_metadata=None,
):
    _validate_intercept_root_metadata(root, dataset_path)

    if modality not in ('rgb', 'event', 'sparse_ball'):
        raise ValueError(f"Unsupported interception modality {modality!r} for {dataset_path}")

    if '/action' not in root:
        raise ValueError(f"Missing required dataset: /action in {dataset_path}")
    if '/observations/qpos' not in root:
        raise ValueError(f"Missing required dataset: /observations/qpos in {dataset_path}")
    image_key = '/observations/sparse_ball' if modality == 'sparse_ball' else (
        '/observations/images/rgb' if modality == 'rgb' else '/observations/images/event')
    if image_key not in root:
        raise ValueError(f"Missing required dataset: {image_key} in {dataset_path}")
    if '/observations/timestamps' not in root:
        raise ValueError(f"Missing required dataset: /observations/timestamps in {dataset_path}")

    action = np.asarray(root['/action'][()], dtype=np.float32)
    qpos = np.asarray(root['/observations/qpos'][()], dtype=np.float32)
    image_ds = root[image_key]
    obs_timestamps = np.asarray(root['/observations/timestamps'][()], dtype=np.float64)

    if action.ndim != 2 or action.shape[1] != 1:
        raise ValueError(
            f"Interception /action must have shape (T,1) in {dataset_path}, got {action.shape}"
        )
    if qpos.ndim != 2 or qpos.shape[1] != 7:
        raise ValueError(
            f"Interception /observations/qpos must have shape (T,7) in {dataset_path}, got {qpos.shape}"
        )

    T = action.shape[0]
    if T < 2:
        raise ValueError(f"Interception episode must have T>=2 in {dataset_path}, got T={T}")
    if qpos.shape[0] != T:
        raise ValueError(
            f"Unequal numbers of qpos and action steps in {dataset_path}: {qpos.shape[0]} != {T}"
        )
    if image_ds.shape[0] != T:
        raise ValueError(
            f"Unequal numbers of visual and action steps in {dataset_path}: {image_ds.shape[0]} != {T}"
        )
    if obs_timestamps.shape != (T,):
        raise ValueError(
            f"Interception /observations/timestamps must have shape (T,) in {dataset_path}, got {obs_timestamps.shape}"
        )

    if not np.isfinite(action).all():
        raise ValueError(f"Non-finite values found in /action for {dataset_path}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"Non-finite values found in /observations/qpos for {dataset_path}")

    if modality == 'sparse_ball':
        if image_ds.shape != (T, SPARSE_BALL_FEATURE_DIM):
            raise ValueError(f"Interception sparse_ball must have shape (T,6), got {image_ds.shape}")
        required = ('input_modality', 'sparse_feature_names', 'sparse_history_offsets',
                    'image_width', 'image_height', 'coordinate_convention', 'velocity_convention',
                    'max_observation_age_sec', 'ball_source_topic', 'source_timestamp_policy')
        required = required + ('missing_observation_policy',)
        missing = [key for key in required if key not in root.attrs]
        if missing:
            raise ValueError(f"Missing sparse_ball metadata in {dataset_path}: {missing}")
        names = [_decode_h5_attr(item) for item in np.asarray(root.attrs['sparse_feature_names']).reshape(-1)]
        if tuple(names) != SPARSE_BALL_FEATURE_NAMES:
            raise ValueError(f"sparse feature order mismatch: {names}")

    event_metadata = None
    if modality == 'event':
        event_array = np.asarray(image_ds[()])
        if event_array.ndim != 4 or event_array.shape[-1] != 3:
            raise ValueError(
                f"Interception /observations/images/event must have shape (T,H,W,3) in {dataset_path}, got {event_array.shape}"
            )
        if event_array.dtype != np.uint8:
            raise ValueError(
                f"Interception /observations/images/event must be uint8 in {dataset_path}, got {event_array.dtype}"
            )

        for required_key in ('/event_source_timestamps', '/event_source_age_sec', '/event_count_per_channel'):
            if required_key not in root:
                raise ValueError(f"Missing required dataset: {required_key} in {dataset_path}")

        event_source_timestamps = np.asarray(root['/event_source_timestamps'][()], dtype=np.float64)
        event_source_age_sec = np.asarray(root['/event_source_age_sec'][()], dtype=np.float64)
        event_count_per_channel = np.asarray(root['/event_count_per_channel'][()])

        if event_source_timestamps.shape != (T,):
            raise ValueError(
                f"Interception /event_source_timestamps must have shape (T,) in {dataset_path}, got {event_source_timestamps.shape}"
            )
        if event_source_age_sec.shape != (T,):
            raise ValueError(
                f"Interception /event_source_age_sec must have shape (T,) in {dataset_path}, got {event_source_age_sec.shape}"
            )
        if event_count_per_channel.shape != (T, 3):
            raise ValueError(
                f"Interception /event_count_per_channel must have shape (T,3) in {dataset_path}, got {event_count_per_channel.shape}"
            )

        if not np.isfinite(event_source_timestamps).all():
            raise ValueError(f"Non-finite values found in /event_source_timestamps for {dataset_path}")
        if not np.isfinite(event_source_age_sec).all():
            raise ValueError(f"Non-finite values found in /event_source_age_sec for {dataset_path}")
        if np.any(event_source_timestamps > obs_timestamps):
            raise ValueError(
                f"Non-causal event source timestamps found in {dataset_path}: source timestamp exceeds observation timestamp"
            )
        if np.any(event_source_age_sec < 0.0):
            raise ValueError(f"Negative event source ages found in {dataset_path}")

        computed_age = obs_timestamps - event_source_timestamps
        if not np.allclose(event_source_age_sec, computed_age, atol=1e-6):
            raise ValueError(
                f"Event source ages mismatch in {dataset_path}: expected observation_timestamp - source_timestamp"
            )

        event_metadata = _extract_intercept_event_metadata(root, dataset_path)
        if expected_event_metadata is not None and event_metadata != expected_event_metadata:
            raise ValueError(
                f"Interception event metadata mismatch across episodes in {dataset_path}: "
                f"expected {expected_event_metadata}, found {event_metadata}"
            )

    return qpos, action, event_metadata

def rotate_n_crop_transform(img, size=(360, 480), angle=None, top=None):
    if angle is None:
        angle = np.random.random() * 10 - 5
    if top is None:
        w, h = img.size
        top_h = np.random.randint(0, max(1, h - size[0] + 1))
        top_w = np.random.randint(0, max(1, w - size[1] + 1))
        top = [top_h, top_w]

    img = transforms.functional.rotate(img, angle)
    img = transforms.functional.crop(img, *top, *size)
    return img


def _is_grayscale_image(img):
    if img.ndim == 2:
        return True
    if img.ndim == 3 and img.shape[-1] == 1:
        return True
    return False


def _is_event_frame(cam_name, img):
    return ('event' in cam_name.lower()) or _is_grayscale_image(img)


def prepare_image(img, target_size=DEFAULT_IMAGE_SIZE, force_rgb=False):
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if img.shape[-1] not in (1, 3):
        raise ValueError(f"Unsupported channel count: {img.shape[-1]} for image shape {img.shape}")

    pil_input = img[..., 0] if img.shape[-1] == 1 else img
    pil_img = transforms.functional.to_pil_image(pil_input)
    pil_img = transforms.functional.resize(pil_img, list(target_size))

    resized_img = np.asarray(pil_img)
    if resized_img.ndim == 2:
        resized_img = resized_img[..., None]

    if force_rgb and resized_img.shape[-1] == 1:
        resized_img = np.repeat(resized_img, 3, axis=-1)

    return resized_img


def ensure_hwc3(image, cam_name):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape for camera '{cam_name}': {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape for camera '{cam_name}': {image.shape}")
    return image


def maybe_resize_for_rgb_event(image, cam_name, camera_names, target_size=DEFAULT_IMAGE_SIZE):
    if camera_names == ["rgb", "event"]:
        target_size = canonical_image_size(target_size)
        if image.shape[:2] != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        if image.shape[:2] != target_size:
            raise ValueError(
                f"RGB+event mode requires {target_size[0]}x{target_size[1]} images. camera={cam_name}, got={image.shape}"
            )
    return image


def _print_image_pipeline_info(camera_names, target_size=DEFAULT_IMAGE_SIZE, event_channel_indices=None):
    camera_list = ', '.join(camera_names)
    event_camera_names = [name for name in camera_names if 'event' in name.lower()]
    target_size = canonical_image_size(target_size)
    target_h, target_w = target_size

    print(f"[INFO] Camera names: {camera_list}")
    if event_camera_names:
        for event_camera_name in event_camera_names:
            print(f"[INFO] Event camera detected: {event_camera_name}")
    else:
        print("[INFO] Event camera detected: none (grayscale auto-detect still enabled)")
    print("[INFO] Event preprocessing enabled: True (camera-name or grayscale auto-detect)")
    print(f"[INFO] Resizing all images to {target_h}x{target_w}")
    if event_channel_indices is not None:
        print(f"[INFO] Event channel selection enabled: indices={event_channel_indices}")
    else:
        print("[INFO] Event frames converted to fake RGB for ResNet compatibility")


def _prepare_camera_image(curr_image, cam_name, camera_names, image_size, event_channel_indices=None):
    curr_image = np.asarray(curr_image)
    if curr_image.ndim == 2:
        curr_image = curr_image[..., None]

    if cam_name == 'event' and event_channel_indices is not None:
        curr_image = curr_image[..., event_channel_indices]
        if curr_image.shape[-1] != len(event_channel_indices):
            raise ValueError(
                f"Event channel selection failed for {cam_name}: got shape {curr_image.shape}"
            )
    else:
        curr_image = ensure_hwc3(curr_image, cam_name)

    if camera_names != ["rgb", "event"]:
        curr_image = cv2.resize(curr_image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)

    curr_image = maybe_resize_for_rgb_event(curr_image, cam_name, camera_names, target_size=image_size)
    if curr_image.ndim == 2:
        curr_image = curr_image[..., None]
    return curr_image


def _sample_shared_color_jitter_params():
    return {
        'brightness': np.random.uniform(0.5, 1.5),
        'contrast': np.random.uniform(0.8, 1.2),
        'saturation': np.random.uniform(0.8, 1.2),
        'hue': np.random.uniform(-0.1, 0.1),
    }


def _apply_shared_color_jitter(img, jitter_params):
    img = transforms.functional.adjust_brightness(img, jitter_params['brightness'])
    img = transforms.functional.adjust_contrast(img, jitter_params['contrast'])
    img = transforms.functional.adjust_saturation(img, jitter_params['saturation'])
    img = transforms.functional.adjust_hue(img, jitter_params['hue'])
    return img


def _sample_spatial_aug_params():
    angle = np.random.random() * 10 - 5
    top_h = np.random.randint(0, 120)
    top_w = np.random.randint(0, 160)
    return angle, (top_h, top_w)


def _prepare_visual_history_frames(
    frames,
    modality,
    camera_names,
    image_size,
    photometric_aug=False,
    spatial_aug=False,
):
    if modality not in ('rgb', 'event'):
        raise ValueError(f"Unsupported visual modality: {modality}")

    if modality == 'event':
        if photometric_aug:
            raise ValueError(
                'photometric_aug is not supported for interception event mode '
                'because event channels are temporal bins, not colors.'
            )
        if spatial_aug:
            raise ValueError(
                'spatial_aug is not supported for interception event mode in this controlled experiment.'
            )
        processed_frames = []
        for frame in frames:
            processed_frame = _prepare_camera_image(
                np.asarray(frame),
                'event',
                camera_names,
                image_size,
                event_channel_indices=None,
            )
            if processed_frame.ndim != 3 or processed_frame.shape[-1] != 3:
                raise ValueError(
                    f"Interception event frame must resolve to HWC3 after preprocessing, got {processed_frame.shape}"
                )
            processed_frames.append(processed_frame)
        return processed_frames

    jitter_params = _sample_shared_color_jitter_params() if photometric_aug else None
    spatial_params = _sample_spatial_aug_params() if spatial_aug else None

    processed_frames = []
    for frame in frames:
        pil_img = transforms.functional.to_pil_image(frame)
        if jitter_params is not None:
            pil_img = _apply_shared_color_jitter(pil_img, jitter_params)
        if spatial_params is not None:
            angle, top = spatial_params
            pil_img = rotate_n_crop_transform(pil_img, [480, 640], angle, top)
        processed_frame = _prepare_camera_image(
            np.asarray(pil_img),
            'rgb',
            camera_names,
            image_size,
            event_channel_indices=None,
        )
        processed_frames.append(processed_frame)

    return processed_frames


def _prepare_rgb_history_frames(frames, camera_names, image_size, photometric_aug=False, spatial_aug=False):
    return _prepare_visual_history_frames(
        frames,
        modality='rgb',
        camera_names=camera_names,
        image_size=image_size,
        photometric_aug=photometric_aug,
        spatial_aug=spatial_aug,
    )


def _natural_episode_sort_key(dataset_path):
    basename = os.path.basename(dataset_path)
    match = re.match(r'^episode_(\d+)\.hdf5$', basename)
    if match is not None:
        return (0, int(match.group(1)), basename)
    return (1, basename)


def _extract_episode_index(dataset_path):
    basename = os.path.basename(dataset_path)
    match = re.match(r'^episode_(\d+)\.hdf5$', basename)
    if match is None:
        return None
    return int(match.group(1))


def _warn_if_episode_indices_noncontiguous(episode_paths):
    episode_indices_by_dir = {}
    for dataset_path in episode_paths:
        episode_index = _extract_episode_index(dataset_path)
        if episode_index is None:
            continue
        dataset_dir = os.path.dirname(dataset_path)
        episode_indices_by_dir.setdefault(dataset_dir, []).append(episode_index)

    for dataset_dir, episode_indices in episode_indices_by_dir.items():
        unique_indices = sorted(set(episode_indices))
        if not unique_indices:
            continue
        expected_indices = list(range(unique_indices[0], unique_indices[-1] + 1))
        if unique_indices != expected_indices:
            print(f"[WARN] Non-contiguous episode indices detected in {dataset_dir}: {unique_indices}")


def collect_episode_paths(dataset_dirs):
    if isinstance(dataset_dirs, str):
        normalized_dataset_dirs = [dataset_dirs]
    else:
        normalized_dataset_dirs = list(dataset_dirs)

    if len(normalized_dataset_dirs) == 0:
        raise ValueError("No dataset directories were provided.")

    episode_paths = []
    for dataset_dir in normalized_dataset_dirs:
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset path is not a directory: {dataset_dir}")
        dataset_episode_paths = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
        if len(dataset_episode_paths) == 0:
            raise FileNotFoundError(f"No episode_*.hdf5 files found in dataset directory: {dataset_dir}")
        dataset_episode_paths.sort(key=_natural_episode_sort_key)
        episode_paths.extend(dataset_episode_paths)

    if len(episode_paths) == 0:
        joined_dirs = ', '.join(normalized_dataset_dirs)
        raise FileNotFoundError(f"No episode_*.hdf5 files found in dataset directories: {joined_dirs}")

    return episode_paths


# =========================
# split-selection helpers
# =========================

def _compute_episode_action_stats_joint(episode_paths, action_indices=None, action_key='/action', command_flag_key=None):
    """
    Per-episode stats for joint/action datasets.
    Uses only action distribution, since this generic loader does not know dx/dy.
    """
    episode_stats = []
    for episode_idx, dataset_path in enumerate(episode_paths):
        with h5py.File(dataset_path, 'r') as root:
            action = _read_action_array(root, action_key=action_key, command_flag_key=command_flag_key)
            if action_indices is not None:
                action = action[:, action_indices]
        action = np.asarray(action, dtype=np.float32)

        ep_stat = {
            "episode_idx": episode_idx,
            "num_steps": int(action.shape[0]),
            "mean_action": action.mean(axis=0),
            "pos_frac_action": (action > 0).mean(axis=0),
        }
        episode_stats.append(ep_stat)
    return episode_stats


def _compute_episode_action_stats_pose(episode_paths):
    """
    Per-episode stats for pose/action datasets.
    """
    episode_stats = []
    for episode_idx, dataset_path in enumerate(episode_paths):
        with h5py.File(dataset_path, 'r') as root:
            action = root['/ee_action_global'][()]
        action = np.asarray(action, dtype=np.float32)

        ep_stat = {
            "episode_idx": episode_idx,
            "num_steps": int(action.shape[0]),
            "mean_action": action.mean(axis=0),
            "pos_frac_action": (action > 0).mean(axis=0),
        }
        episode_stats.append(ep_stat)
    return episode_stats


def _aggregate_episode_stats(episode_stats, indices):
    """
    Weighted aggregation over episodes using episode length as weight.
    """
    if len(indices) == 0:
        raise ValueError("Cannot aggregate empty split.")

    weights = np.array([episode_stats[i]["num_steps"] for i in indices], dtype=np.float64)
    weights = weights / weights.sum()

    mean_action = np.stack([episode_stats[i]["mean_action"] for i in indices], axis=0)
    pos_frac_action = np.stack([episode_stats[i]["pos_frac_action"] for i in indices], axis=0)

    return {
        "mean_action": (weights[:, None] * mean_action).sum(axis=0),
        "pos_frac_action": (weights[:, None] * pos_frac_action).sum(axis=0),
        "num_selected": len(indices),
        "num_steps": int(sum(episode_stats[i]["num_steps"] for i in indices)),
    }


def _compute_global_episode_stats(episode_stats):
    all_indices = list(range(len(episode_stats)))
    return _aggregate_episode_stats(episode_stats, all_indices)


def _score_split(train_stats, val_stats, global_stats):
    """
    Lower is better.

    We want:
    - train close to val
    - both close to global distribution

    Mean mismatch is normalized by a rough scale from global positivity spread.
    """
    mean_scale = 1.0

    train_val_mean_gap = np.mean(np.abs(train_stats["mean_action"] - val_stats["mean_action"])) / mean_scale
    train_val_pos_gap = np.mean(np.abs(train_stats["pos_frac_action"] - val_stats["pos_frac_action"]))

    val_global_mean_gap = np.mean(np.abs(val_stats["mean_action"] - global_stats["mean_action"])) / mean_scale
    val_global_pos_gap = np.mean(np.abs(val_stats["pos_frac_action"] - global_stats["pos_frac_action"]))

    train_global_mean_gap = np.mean(np.abs(train_stats["mean_action"] - global_stats["mean_action"])) / mean_scale
    train_global_pos_gap = np.mean(np.abs(train_stats["pos_frac_action"] - global_stats["pos_frac_action"]))

    # Emphasize val being representative, since checkpoint selection uses val.
    score = (
        1.0 * train_val_mean_gap +
        2.0 * train_val_pos_gap +
        1.5 * val_global_mean_gap +
        2.5 * val_global_pos_gap +
        0.5 * train_global_mean_gap +
        1.0 * train_global_pos_gap
    )
    return float(score)


def _print_split_summary(train_indices, val_indices, episode_stats, header="Chosen split"):
    train_stats = _aggregate_episode_stats(episode_stats, train_indices)
    val_stats = _aggregate_episode_stats(episode_stats, val_indices)
    global_stats = _compute_global_episode_stats(episode_stats)

    print(f"\n===== {header} =====")
    print(f"train episodes: {len(train_indices)} | val episodes: {len(val_indices)}")
    print(f"train steps   : {train_stats['num_steps']} | val steps   : {val_stats['num_steps']}")

    dim = len(global_stats["mean_action"])
    for d in range(dim):
        print(f"\n--- action dim {d} ---")
        print(f"global mean        : {global_stats['mean_action'][d]: .6f}")
        print(f"train mean         : {train_stats['mean_action'][d]: .6f}")
        print(f"val mean           : {val_stats['mean_action'][d]: .6f}")
        print(f"global pos frac    : {global_stats['pos_frac_action'][d]: .3f}")
        print(f"train pos frac     : {train_stats['pos_frac_action'][d]: .3f}")
        print(f"val pos frac       : {val_stats['pos_frac_action'][d]: .3f}")
        print(f"|train-val mean|   : {abs(train_stats['mean_action'][d] - val_stats['mean_action'][d]): .6f}")
        print(f"|train-val posfrac|: {abs(train_stats['pos_frac_action'][d] - val_stats['pos_frac_action'][d]): .3f}")
    print("")


def _choose_balanced_episode_split(
    episode_stats,
    train_ratio=0.8,
    num_trials=100,
    seed=0,
    verbose=True,
):
    """
    Option A:
    Try many random episode-level splits and keep the most balanced one.
    """
    episode_count = len(episode_stats)
    if episode_count < 2:
        raise ValueError("Need at least 2 episodes to create train/val split.")

    num_train = int(train_ratio * episode_count)
    num_train = max(1, min(num_train, episode_count - 1))

    global_stats = _compute_global_episode_stats(episode_stats)
    rng = np.random.RandomState(seed)

    best_score = np.inf
    best_train_indices = None
    best_val_indices = None

    for _ in range(num_trials):
        shuffled = rng.permutation(episode_count)
        train_indices = np.sort(shuffled[:num_train])
        val_indices = np.sort(shuffled[num_train:])

        train_stats = _aggregate_episode_stats(episode_stats, train_indices)
        val_stats = _aggregate_episode_stats(episode_stats, val_indices)
        score = _score_split(train_stats, val_stats, global_stats)

        if score < best_score:
            best_score = score
            best_train_indices = train_indices
            best_val_indices = val_indices

    if verbose:
        print(f"\nBalanced split search: {num_trials} trials, best score = {best_score:.6f}")
        _print_split_summary(best_train_indices, best_val_indices, episode_stats, header="Balanced split summary")

    return best_train_indices, best_val_indices


class EpisodicJointDataset(torch.utils.data.Dataset):
    def __init__(self, episode_paths, camera_names, chunk_size, norm_stats, qpos_indices=None, action_indices=None, action_key='/action', command_flag_key=None, photometric_aug=False, spatial_aug=False, image_size=None, event_channel_indices=None, rgb_history_frames=1):
        super(EpisodicJointDataset).__init__()
        self.qpos_indices = qpos_indices
        self.action_indices = action_indices
        self.action_key = action_key
        self.command_flag_key = command_flag_key
        self.episode_paths = list(episode_paths)
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.photometric_aug = photometric_aug
        self.spatial_aug = spatial_aug
        self.is_sim = None
        self._printed_image_debug = False
        self._printed_temporal_debug = False
        self.image_size = canonical_image_size(image_size)
        self.event_channel_indices = event_channel_indices
        self.rgb_history_frames = int(rgb_history_frames)
        if self.rgb_history_frames not in (1, 2, 3):
            raise ValueError(f"rgb_history_frames must be one of 1, 2, or 3, got {self.rgb_history_frames}")
        if self.rgb_history_frames > 1:
            if self.camera_names != ['rgb']:
                raise ValueError(
                    f"rgb_history_frames={self.rgb_history_frames} currently requires camera_names=['rgb'], got {self.camera_names}"
                )
            if self.event_channel_indices is not None:
                raise ValueError(
                    'rgb_history_frames > 1 does not support event_channel_selection or event_channel_indices.'
                )
        _print_image_pipeline_info(
            self.camera_names,
            target_size=self.image_size,
            event_channel_indices=self.event_channel_indices,
        )
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_paths)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        dataset_path = self.episode_paths[index]
        with h5py.File(dataset_path, 'r') as root:
            is_sim = bool(root.attrs.get('sim', False))
            action_full = _read_action_array(
                root,
                action_key=self.action_key,
                command_flag_key=self.command_flag_key,
            )
            episode_len = action_full.shape[0]
            temporal_indices = None
            if self.rgb_history_frames > 1:
                if episode_len < self.rgb_history_frames:
                    raise ValueError(
                        f"RGB temporal history requires at least {self.rgb_history_frames} frames in {dataset_path}, found episode length {episode_len}"
                    )
                history_start = self.rgb_history_frames - 1
                if sample_full_episode:
                    start_ts = history_start
                else:
                    start_ts = np.random.randint(history_start, episode_len)
                temporal_indices = list(range(start_ts - self.rgb_history_frames + 1, start_ts + 1))
            else:
                if sample_full_episode:
                    start_ts = 0
                else:
                    start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            if self.qpos_indices is not None:
                qpos = qpos[self.qpos_indices]
            image_dict = dict()
            if self.rgb_history_frames > 1:
                frames = [root['/observations/images/rgb'][ts] for ts in temporal_indices]
                processed_frames = _prepare_rgb_history_frames(
                    frames,
                    self.camera_names,
                    self.image_size,
                    photometric_aug=self.photometric_aug,
                    spatial_aug=self.spatial_aug,
                )
                image_dict['rgb'] = np.concatenate(processed_frames, axis=-1)
            else:
                for cam_name in self.camera_names:
                    if cam_name.endswith('stereo'):
                        left_img = root[f'/observations/images/{cam_name[:-6]}left'][start_ts]
                        right_img = root[f'/observations/images/{cam_name[:-6]}right'][start_ts]
                        left_is_event = _is_event_frame(cam_name, left_img)
                        right_is_event = _is_event_frame(cam_name, right_img)
                        left_img = transforms.functional.to_pil_image(left_img)
                        right_img = transforms.functional.to_pil_image(right_img)
                        if self.spatial_aug:
                            angle = np.random.random() * 10 - 5
                            top_h = np.random.randint(0, 120)
                            top_w = np.random.randint(0, 160)
                        if self.photometric_aug and not left_is_event:
                            left_img = color_transform(left_img)
                        if self.spatial_aug:
                            left_img = rotate_n_crop_transform(left_img, [480, 640], angle, (top_h, top_w))
                        if self.photometric_aug and not right_is_event:
                            right_img = color_transform(right_img)
                        if self.spatial_aug:
                            right_img = rotate_n_crop_transform(right_img, [480, 640], angle, (top_h, top_w))
                        left_img = np.asarray(left_img)
                        right_img = np.asarray(right_img)
                        left_img = ensure_hwc3(left_img, f'{cam_name}_left')
                        right_img = ensure_hwc3(right_img, f'{cam_name}_right')
                        left_img = maybe_resize_for_rgb_event(left_img, f'{cam_name}_left', self.camera_names, target_size=self.image_size)
                        right_img = maybe_resize_for_rgb_event(right_img, f'{cam_name}_right', self.camera_names, target_size=self.image_size)
                        stereo_img = np.concatenate([left_img, right_img], axis=1) # width dimension
                        if self.camera_names != ["rgb", "event"]:
                            stereo_img = cv2.resize(
                                stereo_img,
                                (self.image_size[1], self.image_size[0]),
                                interpolation=cv2.INTER_AREA,
                            )
                        image_dict[cam_name] = ensure_hwc3(stereo_img, cam_name)
                    else:
                        img = root[f'/observations/images/{cam_name}'][start_ts]
                        is_event = _is_event_frame(cam_name, img)
                        img = transforms.functional.to_pil_image(img)
                        if self.photometric_aug and not is_event:
                            img = color_transform(img)
                        if self.spatial_aug:
                            img = rotate_n_crop_transform(img)
                        img = np.asarray(img)
                        img = _prepare_camera_image(
                            img,
                            cam_name,
                            self.camera_names,
                            self.image_size,
                            event_channel_indices=self.event_channel_indices,
                        )
                        image_dict[cam_name] = img
            # get all actions after and including start_ts
            action = action_full[start_ts:min(start_ts+self.chunk_size, episode_len)]
            if self.action_indices is not None:
                action = action[:, self.action_indices]
            action_len, action_dof = action.shape

        self.is_sim = is_sim
        padded_action = np.zeros((self.chunk_size, action_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras in exact camera_names order
        all_cam_images = []
        for cam_name in self.camera_names:
            cam_image = np.asarray(image_dict[cam_name])
            if cam_image.ndim == 2:
                cam_image = cam_image[..., None]
            cam_image = np.transpose(cam_image, (2, 0, 1))
            all_cam_images.append(cam_image)
        image_data = torch.from_numpy(np.stack(all_cam_images, axis=0))
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        assert image_data.ndim == 4, image_data.shape
        assert image_data.shape[0] == len(self.camera_names), image_data.shape
        if self.rgb_history_frames > 1:
            assert self.camera_names == ['rgb'], self.camera_names
            assert image_data.shape == (1, 3 * self.rgb_history_frames, self.image_size[0], self.image_size[1]), image_data.shape
        elif self.event_channel_indices is None:
            assert image_data.shape[1] == 3, image_data.shape
        else:
            assert self.camera_names == ['event'], self.camera_names
            assert image_data.shape[1] == len(self.event_channel_indices), image_data.shape
        if self.camera_names == ["rgb", "event"]:
            assert image_data.shape == (2, 3, self.image_size[0], self.image_size[1]), image_data.shape
        if self.rgb_history_frames > 1:
            if not self._printed_temporal_debug:
                print(
                    f"[DEBUG] rgb_history_frames={self.rgb_history_frames}, "
                    f"temporal_indices={temporal_indices}, final_image_data.shape={tuple(image_data.shape)}"
                )
                self._printed_temporal_debug = True
                self._printed_image_debug = True
        elif not self._printed_image_debug:
            print(f"[DEBUG] camera_names={self.camera_names}, image_data.shape={tuple(image_data.shape)}")
            self._printed_image_debug = True

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


class EpisodicInterceptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_paths,
        camera_names,
        chunk_size,
        norm_stats,
        history_offsets=INTERCEPT_HISTORY_OFFSETS_DEFAULT,
        photometric_aug=False,
        spatial_aug=False,
        image_size=None,
        input_modality='rgb',
        expected_event_metadata=None,
    ):
        super(EpisodicInterceptDataset).__init__()
        self.episode_paths = list(episode_paths)
        self.camera_names = list(camera_names)
        self.chunk_size = int(chunk_size)
        self.norm_stats = norm_stats
        self.history_offsets = tuple(int(offset) for offset in history_offsets)
        self.photometric_aug = bool(photometric_aug)
        self.spatial_aug = bool(spatial_aug)
        self.image_size = canonical_image_size(image_size)
        self.input_modality = str(input_modality)
        self.expected_event_metadata = expected_event_metadata
        self.is_sim = None
        self._printed_image_debug = False
        self._printed_intercept_debug = False

        if self.input_modality not in ('rgb', 'event', 'sparse_ball'):
            raise ValueError(
                f"Interception input_modality must be 'rgb' or 'event', got {self.input_modality!r}"
            )
        expected_camera_names = {'rgb': ['rgb'], 'event': ['event'], 'sparse_ball': ['sparse_ball']}[self.input_modality]
        if self.camera_names != expected_camera_names:
            raise ValueError(
                f"Interception dataset requires camera_names={expected_camera_names} for modality {self.input_modality!r}, got {self.camera_names}"
            )
        if len(self.history_offsets) != 3:
            raise ValueError(
                f"Interception requires exactly 3 history offsets, got {self.history_offsets}"
            )

        if self.input_modality != 'sparse_ball':
            _print_image_pipeline_info(self.camera_names, target_size=self.image_size)
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_paths)

    def __getitem__(self, index):
        dataset_path = self.episode_paths[index]
        with h5py.File(dataset_path, 'r') as root:
            is_sim = bool(root.attrs.get('sim', False))
            qpos_full, action_full, _ = _validate_intercept_episode_structure(
                root,
                dataset_path,
                modality=self.input_modality,
                expected_event_metadata=self.expected_event_metadata,
            )
            T = action_full.shape[0]

            # Valid anchors are 0..T-2 so token-0 always has a true future s(t+1)-s(t).
            anchor_t = np.random.randint(0, T - 1)
            history_indices = compute_history_indices(anchor_t, self.history_offsets)

            qpos_history = qpos_full[history_indices]
            qpos = qpos_history.reshape(-1).astype(np.float32)

            if self.input_modality == 'sparse_ball':
                sparse_history = np.stack(
                    [np.asarray(root['/observations/sparse_ball'][ts], dtype=np.float32) for ts in history_indices]
                )
            else:
                visual_key = '/observations/images/rgb' if self.input_modality == 'rgb' else '/observations/images/event'
                visual_frames = [root[visual_key][ts] for ts in history_indices]
                processed_frames = _prepare_visual_history_frames(
                    visual_frames, self.input_modality, self.camera_names, self.image_size,
                    photometric_aug=self.photometric_aug, spatial_aug=self.spatial_aug)
                image_rgb = np.concatenate(processed_frames, axis=-1)

            anchor_s = float(action_full[anchor_t, 0])
            future_end = min(T, anchor_t + 1 + self.chunk_size)
            future_abs = action_full[anchor_t + 1:future_end, 0:1]
            action = future_abs - anchor_s
            action_len = action.shape[0]

        self.is_sim = is_sim

        padded_action = np.zeros((self.chunk_size, 1), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size, dtype=np.float32)
        is_pad[action_len:] = 1.0

        if self.input_modality == 'sparse_ball':
            image_data = torch.from_numpy(sparse_history).float()
        else:
            image_data = torch.from_numpy(np.transpose(image_rgb, (2, 0, 1))[None, ...])
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        expected_input_shape = (3, 6) if self.input_modality == 'sparse_ball' else (1, 9, self.image_size[0], self.image_size[1])
        assert image_data.shape == expected_input_shape, image_data.shape
        assert qpos_data.shape == (21,), qpos_data.shape
        assert action_data.shape == (self.chunk_size, 1), action_data.shape
        assert is_pad.shape == (self.chunk_size,), is_pad.shape

        if not self._printed_intercept_debug:
            first_future_count = min(5, action_len)
            print(
                f"[DEBUG] intercept sample anchor_t={anchor_t}, history_indices={history_indices}, "
                f"anchor_abs_s={anchor_s:+.6f}, "
                f"future_abs_s[:{first_future_count}]={future_abs[:first_future_count, 0].tolist()}, "
                f"delta_s[:{first_future_count}]={action[:first_future_count, 0].tolist()}, "
                f"image_shape={tuple(image_data.shape)}, qpos_shape={tuple(qpos_data.shape)}, "
                f"action_shape={tuple(action_data.shape)}"
            )
            self._printed_intercept_debug = True
            self._printed_image_debug = True

        if self.input_modality == 'sparse_ball':
            pass  # ACTPolicy applies the saved sparse statistics for both training and rollout.
        else:
            image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        qpos_data = (qpos_data - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']

        # Keep padded tokens neutral even after normalization.
        action_data[is_pad] = 0.0

        return image_data, qpos_data, action_data, is_pad


def get_joint_norm_stats(episode_paths, qpos_indices=None, action_indices=None, action_key='/action', command_flag_key=None):
    all_qpos_data = []
    all_action_data = []

    for dataset_path in episode_paths:
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = _read_action_array(root, action_key=action_key, command_flag_key=command_flag_key)

        if qpos_indices is not None:
            qpos = qpos[:, qpos_indices]
        if action_indices is not None:
            action = action[:, action_indices]

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    qpos_mean = all_qpos_data.mean(dim=0).numpy()
    qpos_std = all_qpos_data.std(dim=0).numpy().clip(1e-2, np.inf)

    action_mean = all_action_data.mean(dim=0).numpy()
    action_std = all_action_data.std(dim=0).numpy().clip(1e-2, np.inf)
    if command_flag_key is not None:
        action_mean[-1] = 0.0
        action_std[-1] = 1.0

    stats = {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "example_qpos": qpos,
    }

    return stats


def get_intercept_norm_stats(
    episode_paths,
    chunk_size,
    history_offsets=INTERCEPT_HISTORY_OFFSETS_DEFAULT,
    input_modality='rgb',
    event_metadata=None,
):
    qpos_histories = []
    sparse_histories = []
    sparse_checkpoint_metadata = None
    delta_tokens = []

    for dataset_path in episode_paths:
        with h5py.File(dataset_path, 'r') as root:
            qpos_full, action_full, _ = _validate_intercept_episode_structure(
                root,
                dataset_path,
                modality=input_modality,
                expected_event_metadata=event_metadata,
            )
            sparse_full = (np.asarray(root['/observations/sparse_ball'][()], dtype=np.float32)
                           if input_modality == 'sparse_ball' else None)
            if input_modality == 'sparse_ball' and sparse_checkpoint_metadata is None:
                sparse_checkpoint_metadata = {
                    key: _decode_h5_attr(root.attrs[key])
                    for key in ('image_width', 'image_height', 'coordinate_convention', 'velocity_convention',
                                'max_observation_age_sec', 'ball_source_topic', 'source_timestamp_policy',
                                'missing_observation_policy')
                }

        T = action_full.shape[0]
        for anchor_t in range(T - 1):
            history_indices = compute_history_indices(anchor_t, history_offsets)
            qpos_histories.append(qpos_full[history_indices].reshape(-1))
            if input_modality == 'sparse_ball':
                sparse_histories.append(sparse_full[history_indices])

            future_end = min(T, anchor_t + 1 + chunk_size)
            future_abs = action_full[anchor_t + 1:future_end, 0]
            delta_values = future_abs - action_full[anchor_t, 0]
            if delta_values.size > 0:
                delta_tokens.append(delta_values)

    if len(qpos_histories) == 0:
        raise ValueError('No valid interception anchors found while computing normalization stats.')
    if len(delta_tokens) == 0:
        raise ValueError('No valid interception future-delta tokens found while computing normalization stats.')

    qpos_histories = np.asarray(qpos_histories, dtype=np.float32)
    all_delta_values = np.concatenate(delta_tokens, axis=0).astype(np.float32)

    qpos_mean = qpos_histories.mean(axis=0)
    qpos_std = qpos_histories.std(axis=0).clip(1e-2, np.inf)

    action_mean = np.asarray([all_delta_values.mean()], dtype=np.float32)
    action_std = np.asarray([all_delta_values.std()], dtype=np.float32).clip(1e-2, np.inf)

    stats = {
        'action_mean': action_mean,
        'action_std': action_std,
        'qpos_mean': qpos_mean,
        'qpos_std': qpos_std,
        'example_qpos': qpos_histories[0],
        'data_mode': 'intercept',
        'raw_qpos_dim': 7,
        'state_dim': 21,
        'action_dim': 1,
        'input_modality': str(input_modality),
        'visual_history_frames': 3,
        'visual_history_offsets': list(history_offsets),
        'visual_frame_order': 'oldest_to_newest',
        'channels_per_visual_frame': 3,
        'rgb_history_frames': 3,
        'rgb_history_offsets': list(history_offsets),
        'rgb_frame_order': 'oldest_to_newest',
        'qpos_history_offsets': list(history_offsets),
        'qpos_flatten_order': 'oldest_to_newest',
        'image_channels': 9,
        'image_normalization': 'imagenet' if str(input_modality) == 'rgb' else 'shifted_3chef_centered',
        'action_type': 'measured_tcp_s_delta',
        'action_representation': 'future_delta_relative_to_anchor',
        'action_anchor_offset': 0,
        'action_first_target_offset': 1,
        'action_positive_direction': 'robot_base_positive_x',
        'action_units': 'm',
        'camera_names': {'rgb': ['rgb'], 'event': ['event'], 'sparse_ball': ['sparse_ball']}[str(input_modality)],
    }

    if str(input_modality) == 'sparse_ball':
        values = np.concatenate(sparse_histories, axis=0)
        stats['sparse_mean'] = values.mean(axis=0).astype(np.float32)
        stats['sparse_std'] = values.std(axis=0).clip(1e-2, np.inf).astype(np.float32)
        stats['sparse_feature_dim'] = 6
        stats['sparse_feature_names'] = list(SPARSE_BALL_FEATURE_NAMES)
        stats['sparse_history_offsets'] = list(history_offsets)
        stats['sparse_history_length'] = 3
        stats['image_channels'] = 0
        stats['image_normalization'] = 'none'
        stats.update(sparse_checkpoint_metadata)

    if str(input_modality) == 'event':
        if event_metadata is None:
            raise ValueError('event_metadata is required when input_modality=event')
        for key in (
            'event_representation',
            'event_frame_mode',
            'event_frame_windows_ms',
            'event_channel_order',
            'event_scaling',
            'event_clip_count',
            'event_neutral_u8',
            'event_sampling_policy',
        ):
            if key not in event_metadata:
                raise ValueError(
                    f"Missing canonical event metadata key for interception stats: {key}"
                )
            stats[key] = event_metadata[key]

    return stats


def load_joint_data(
    dataset_dirs,
    camera_names,
    chunk_size,
    batch_size_train,
    batch_size_val,
    model_dof=None,
    photometric_aug=False,
    spatial_aug=False,
    split_num_trials=500,
    split_seed=0,
    qpos_dim=None,
    action_dim=None,
    qpos_indices=None,
    action_indices=None,
    action_key='/action',
    command_flag_key=None,
    image_size=None,
    event_channel_indices=None,
    rgb_history_frames=1,
):
    rgb_history_frames = int(rgb_history_frames)
    if event_channel_indices is not None and camera_names != ['event']:
        raise NotImplementedError(
            '--event_channel_selection is currently only implemented for --camera_names event'
        )
    if rgb_history_frames > 1:
        if camera_names != ['rgb']:
            raise ValueError(
                f"rgb_history_frames={rgb_history_frames} currently requires camera_names=['rgb'], got {camera_names}"
            )
        if event_channel_indices is not None:
            raise ValueError('rgb_history_frames > 1 does not support event_channel_selection.')

    # Backward-compatible default: model_dof applies to both if explicit dims are not provided.
    if qpos_dim is None and model_dof is not None:
        qpos_dim = model_dof
    if action_dim is None and model_dof is not None:
        action_dim = model_dof

    joint_data_cfg = _build_joint_data_config(
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        qpos_indices=qpos_indices,
        action_indices=action_indices,
        action_key=action_key,
        command_flag_key=command_flag_key,
    )

    episode_paths = collect_episode_paths(dataset_dirs)
    total_episode_count = len(episode_paths)
    if total_episode_count < 2:
        raise ValueError("Need at least 2 episodes for train/val split")

    if isinstance(dataset_dirs, str):
        source_dataset_dirs = [dataset_dirs]
    else:
        source_dataset_dirs = list(dataset_dirs)
    
    print('\nSource dataset dirs:')
    for source_dataset_dir in source_dataset_dirs:
        print(source_dataset_dir)
    print(f'Total discovered episodes: {total_episode_count}')
    print('First few episode paths:')
    for dataset_path in episode_paths[:min(5, total_episode_count)]:
        print(dataset_path)
    print('')
    _warn_if_episode_indices_noncontiguous(episode_paths)

    for dataset_path in episode_paths:
        with h5py.File(dataset_path, 'r') as root:
            _, action = _validate_joint_episode_structure(
                root,
                dataset_path,
                camera_names,
                qpos_dim=joint_data_cfg['qpos_dim'],
                action_dim=joint_data_cfg['action_dim'],
                action_key=joint_data_cfg['action_key'],
                command_flag_key=joint_data_cfg['command_flag_key'],
            )
            if rgb_history_frames > 1 and action.shape[0] < rgb_history_frames:
                raise ValueError(
                    f"RGB temporal history requires at least {rgb_history_frames} frames in {dataset_path}, found episode length {action.shape[0]}"
                )

    # obtain train/val split using balanced random search
    # train_ratio = 0.8
    # episode_stats = _compute_episode_action_stats_joint(
    #     episode_paths,
    #     action_indices=joint_data_cfg['action_indices'],
    #     action_key=joint_data_cfg['action_key'],
    # )
    # train_indices, val_indices = _choose_balanced_episode_split(
    #     episode_stats=episode_stats,
    #     train_ratio=train_ratio,
    #     num_trials=split_num_trials,
    #     seed=split_seed,
    #     verbose=True,
    # )
    episode_indices = list(range(total_episode_count))
    rng = np.random.RandomState(split_seed)
    rng.shuffle(episode_indices)

    split_idx = int(0.8 * total_episode_count)
    split_idx = max(1, min(split_idx, total_episode_count - 1))

    train_indices = episode_indices[:split_idx]
    val_indices = episode_indices[split_idx:]

    print(f'Train episode count: {len(train_indices)}')
    print(f'Val episode count: {len(val_indices)}')
    print(f'[INFO] Augmentation (train): photometric_aug={photometric_aug}, spatial_aug={spatial_aug}')
    print('[INFO] Augmentation (val): photometric_aug=False, spatial_aug=False')

    # obtain normalization stats for qpos and action
    norm_stats = get_joint_norm_stats(
        episode_paths,
        qpos_indices=joint_data_cfg['qpos_indices'],
        action_indices=joint_data_cfg['action_indices'],
        action_key=joint_data_cfg['action_key'],
        command_flag_key=joint_data_cfg['command_flag_key'],
    )

    # construct dataset and dataloader
    train_dataset = EpisodicJointDataset(
        [episode_paths[index] for index in train_indices],
        camera_names,
        chunk_size,
        norm_stats,
        qpos_indices=joint_data_cfg['qpos_indices'],
        action_indices=joint_data_cfg['action_indices'],
        action_key=joint_data_cfg['action_key'],
        command_flag_key=joint_data_cfg['command_flag_key'],
        photometric_aug=photometric_aug,
        spatial_aug=spatial_aug,
        image_size=image_size,
        event_channel_indices=event_channel_indices,
        rgb_history_frames=rgb_history_frames,
    )
    val_dataset = EpisodicJointDataset(
        [episode_paths[index] for index in val_indices],
        camera_names,
        chunk_size,
        norm_stats,
        qpos_indices=joint_data_cfg['qpos_indices'],
        action_indices=joint_data_cfg['action_indices'],
        action_key=joint_data_cfg['action_key'],
        command_flag_key=joint_data_cfg['command_flag_key'],
        photometric_aug=False,
        spatial_aug=False,
        image_size=image_size,
        event_channel_indices=event_channel_indices,
        rgb_history_frames=rgb_history_frames,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def load_intercept_data(
    dataset_dirs,
    camera_names,
    chunk_size,
    batch_size_train,
    batch_size_val,
    photometric_aug=False,
    spatial_aug=False,
    split_num_trials=500,
    split_seed=0,
    raw_qpos_dim=7,
    state_dim=21,
    action_dim=1,
    image_size=None,
    rgb_history_frames=3,
    visual_history_frames=None,
    history_offsets=INTERCEPT_HISTORY_OFFSETS_DEFAULT,
    input_modality=None,
):
    del split_num_trials  # currently unused in interception loader

    if camera_names not in (['rgb'], ['event'], ['sparse_ball']):
        raise ValueError(
            f"Interception requires camera_names=['rgb'] or ['event'], got {camera_names}"
        )
    inferred_modality = {'rgb': 'rgb', 'event': 'event', 'sparse_ball': 'sparse_ball'}[camera_names[0]]
    if input_modality is not None and str(input_modality) != inferred_modality:
        raise ValueError(f"input_modality={input_modality!r} conflicts with camera_names={camera_names}")
    input_modality = inferred_modality
    if int(raw_qpos_dim) != 7:
        raise ValueError(f"Interception raw_qpos_dim must be 7, got {raw_qpos_dim}")
    if int(state_dim) != 21:
        raise ValueError(f"Interception state_dim must be 21, got {state_dim}")
    if int(action_dim) != 1:
        raise ValueError(f"Interception action_dim must be 1, got {action_dim}")

    resolved_visual_history_frames = (
        int(rgb_history_frames) if visual_history_frames is None else int(visual_history_frames)
    )
    if int(rgb_history_frames) != resolved_visual_history_frames:
        raise ValueError(
            f"Interception visual-history conflict: rgb_history_frames={rgb_history_frames} "
            f"but visual_history_frames={visual_history_frames}"
        )
    if resolved_visual_history_frames != 3:
        raise ValueError(
            f"Interception visual_history_frames must be 3, got {resolved_visual_history_frames}"
        )

    history_offsets = tuple(int(offset) for offset in history_offsets)
    if history_offsets != INTERCEPT_HISTORY_OFFSETS_DEFAULT:
        raise ValueError(
            f"Interception history_offsets must be {INTERCEPT_HISTORY_OFFSETS_DEFAULT}, got {history_offsets}"
        )

    episode_paths = collect_episode_paths(dataset_dirs)
    total_episode_count = len(episode_paths)
    if total_episode_count < 2:
        raise ValueError('Need at least 2 episodes for train/val split')

    if isinstance(dataset_dirs, str):
        source_dataset_dirs = [dataset_dirs]
    else:
        source_dataset_dirs = list(dataset_dirs)

    print('\nSource dataset dirs:')
    for source_dataset_dir in source_dataset_dirs:
        print(source_dataset_dir)
    print(f'Total discovered episodes: {total_episode_count}')
    print('First few episode paths:')
    for dataset_path in episode_paths[:min(5, total_episode_count)]:
        print(dataset_path)
    print('')
    _warn_if_episode_indices_noncontiguous(episode_paths)

    expected_event_metadata = None
    for dataset_path in episode_paths:
        with h5py.File(dataset_path, 'r') as root:
            _, _, event_metadata = _validate_intercept_episode_structure(
                root,
                dataset_path,
                modality=input_modality,
                expected_event_metadata=expected_event_metadata,
            )
            if input_modality == 'event' and expected_event_metadata is None:
                expected_event_metadata = event_metadata

    episode_indices = list(range(total_episode_count))
    rng = np.random.RandomState(split_seed)
    rng.shuffle(episode_indices)

    split_idx = int(0.8 * total_episode_count)
    split_idx = max(1, min(split_idx, total_episode_count - 1))

    train_indices = episode_indices[:split_idx]
    val_indices = episode_indices[split_idx:]

    print(f'Train episode count: {len(train_indices)}')
    print(f'Val episode count: {len(val_indices)}')
    print(f'[INFO] Augmentation (train): photometric_aug={photometric_aug}, spatial_aug={spatial_aug}')
    print('[INFO] Augmentation (val): photometric_aug=False, spatial_aug=False')

    train_episode_paths = [episode_paths[index] for index in train_indices]
    val_episode_paths = [episode_paths[index] for index in val_indices]

    # Interception normalization is derived from train split only:
    # - qpos: flattened 3x7 histories at valid anchors
    # - action: non-padded future delta tokens s(t+k+1)-s(t)
    norm_stats = get_intercept_norm_stats(
        train_episode_paths,
        chunk_size=chunk_size,
        history_offsets=history_offsets,
        input_modality=input_modality,
        event_metadata=expected_event_metadata,
    )

    train_dataset = EpisodicInterceptDataset(
        train_episode_paths,
        camera_names,
        chunk_size,
        norm_stats,
        history_offsets=history_offsets,
        photometric_aug=photometric_aug,
        spatial_aug=spatial_aug,
        image_size=image_size,
        input_modality=input_modality,
        expected_event_metadata=expected_event_metadata,
    )
    val_dataset = EpisodicInterceptDataset(
        val_episode_paths,
        camera_names,
        chunk_size,
        norm_stats,
        history_offsets=history_offsets,
        photometric_aug=False,
        spatial_aug=False,
        image_size=image_size,
        input_modality=input_modality,
        expected_event_metadata=expected_event_metadata,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


class EpisodicPoseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_paths, camera_names, chunk_size, norm_stats, photometric_aug=False, spatial_aug=False, image_size=None, event_channel_indices=None):
        super(EpisodicPoseDataset).__init__()
        self.episode_paths = list(episode_paths)
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.photometric_aug = photometric_aug
        self.spatial_aug = spatial_aug
        self.is_sim = None
        self._printed_image_debug = False
        self.image_size = canonical_image_size(image_size)
        self.event_channel_indices = event_channel_indices
        _print_image_pipeline_info(
            self.camera_names,
            target_size=self.image_size,
            event_channel_indices=self.event_channel_indices,
        )
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_paths)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        dataset_path = self.episode_paths[index]
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            episode_len = root['/observations/ee_pose_global'].shape[0] - 120  # hardcode for TOCABI data, do not train moving to ready pose
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/ee_pose_global'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name.endswith('stereo'):
                    left_img = root[f'/observations/images/{cam_name[:-6]}left'][start_ts]
                    right_img = root[f'/observations/images/{cam_name[:-6]}right'][start_ts]
                    left_is_event = _is_event_frame(cam_name, left_img)
                    right_is_event = _is_event_frame(cam_name, right_img)
                    left_img = transforms.functional.to_pil_image(left_img)
                    right_img = transforms.functional.to_pil_image(right_img)
                    if self.spatial_aug:
                        angle = np.random.random() * 10 - 5
                        top_h = np.random.randint(0, 120)
                        top_w = np.random.randint(0, 160)
                    if self.photometric_aug and not left_is_event:
                        left_img = color_transform(left_img)
                    if self.spatial_aug:
                        left_img = rotate_n_crop_transform(left_img, [480, 640], angle, (top_h, top_w))
                    if self.photometric_aug and not right_is_event:
                        right_img = color_transform(right_img)
                    if self.spatial_aug:
                        right_img = rotate_n_crop_transform(right_img, [480, 640], angle, (top_h, top_w))
                    left_img = np.asarray(left_img)
                    right_img = np.asarray(right_img)
                    left_img = ensure_hwc3(left_img, f'{cam_name}_left')
                    right_img = ensure_hwc3(right_img, f'{cam_name}_right')
                    left_img = maybe_resize_for_rgb_event(left_img, f'{cam_name}_left', self.camera_names, target_size=self.image_size)
                    right_img = maybe_resize_for_rgb_event(right_img, f'{cam_name}_right', self.camera_names, target_size=self.image_size)
                    stereo_img = np.concatenate([left_img, right_img], axis=1) # width dimension
                    if self.camera_names != ["rgb", "event"]:
                        stereo_img = cv2.resize(
                            stereo_img,
                            (self.image_size[1], self.image_size[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    image_dict[cam_name] = ensure_hwc3(stereo_img, cam_name)
                else:
                    img = root[f'/observations/images/{cam_name}'][start_ts]
                    is_event = _is_event_frame(cam_name, img)
                    img = transforms.functional.to_pil_image(img)
                    if self.photometric_aug and not is_event:
                        img = color_transform(img)
                    if self.spatial_aug:
                        img = rotate_n_crop_transform(img)
                    img = np.asarray(img)
                    img = _prepare_camera_image(
                        img,
                        cam_name,
                        self.camera_names,
                        self.image_size,
                        event_channel_indices=self.event_channel_indices,
                    )
                    image_dict[cam_name] = img
            # get all actions after and including start_ts
            action = root['/ee_action_global'][start_ts:min(start_ts+self.chunk_size, episode_len)]
            action_len, action_dof = action.shape

        self.is_sim = is_sim
        padded_action = np.zeros((self.chunk_size, action_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras in exact camera_names order
        all_cam_images = []
        for cam_name in self.camera_names:
            cam_image = np.asarray(image_dict[cam_name])
            if cam_image.ndim == 2:
                cam_image = cam_image[..., None]
            cam_image = np.transpose(cam_image, (2, 0, 1))
            all_cam_images.append(cam_image)
        image_data = torch.from_numpy(np.stack(all_cam_images, axis=0))
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        assert image_data.ndim == 4, image_data.shape
        assert image_data.shape[0] == len(self.camera_names), image_data.shape
        if self.event_channel_indices is None:
            assert image_data.shape[1] == 3, image_data.shape
        else:
            assert self.camera_names == ['event'], self.camera_names
            assert image_data.shape[1] == len(self.event_channel_indices), image_data.shape
        if self.camera_names == ["rgb", "event"]:
            assert image_data.shape == (2, 3, self.image_size[0], self.image_size[1]), image_data.shape
        if not self._printed_image_debug:
            print(f"[DEBUG] camera_names={self.camera_names}, image_data.shape={tuple(image_data.shape)}")
            self._printed_image_debug = True

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_pose_norm_stats(episode_paths):
    all_qpos_data = []
    all_action_data = []
    for dataset_path in episode_paths:
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/ee_pose_global'][()]
            action = root['/ee_action_global'][()]
        all_qpos_data.append(torch.from_numpy(qpos[:,9:12])) # do not normalize 9D roation & binary gripper state
        all_action_data.append(torch.from_numpy(action[:,9:12]))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = np.zeros(13, dtype=np.float32)
    action_mean[9:12] = all_action_data.mean(dim=0).numpy()
    action_std = np.ones(13, dtype=np.float32)
    action_std[9:12] = all_action_data.std(dim=0).numpy()
    action_std = action_std.clip(1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = np.zeros(13, dtype=np.float32)
    qpos_mean[9:12] = all_qpos_data.mean(dim=0).numpy()
    qpos_std = np.ones(13, dtype=np.float32)
    qpos_std[9:12] = all_qpos_data.std(dim=0).numpy()
    qpos_std = qpos_std.clip(1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean, "action_std": action_std,
             "qpos_mean": qpos_mean, "qpos_std": qpos_std,
             "example_qpos": qpos}

    return stats


def load_pose_data(
    dataset_dirs,
    camera_names,
    chunk_size,
    batch_size_train,
    batch_size_val,
    photometric_aug=False,
    spatial_aug=False,
    split_num_trials=100,
    split_seed=0,
    image_size=None,
    event_channel_indices=None,
):
    if event_channel_indices is not None and camera_names != ['event']:
        raise NotImplementedError(
            '--event_channel_selection is currently only implemented for --camera_names event'
        )

    episode_paths = collect_episode_paths(dataset_dirs)
    total_episode_count = len(episode_paths)
    if total_episode_count < 2:
        raise ValueError("Need at least 2 episodes for train/val split")

    if isinstance(dataset_dirs, str):
        source_dataset_dirs = [dataset_dirs]
    else:
        source_dataset_dirs = list(dataset_dirs)

    print('\nSource dataset dirs:')
    for source_dataset_dir in source_dataset_dirs:
        print(source_dataset_dir)
    print(f'Total discovered episodes: {total_episode_count}')
    print('First few episode paths:')
    for dataset_path in episode_paths[:min(5, total_episode_count)]:
        print(dataset_path)
    print('')
    _warn_if_episode_indices_noncontiguous(episode_paths)

    # obtain train/val split using balanced random search
    train_ratio = 0.8
    episode_stats = _compute_episode_action_stats_pose(episode_paths)
    train_indices, val_indices = _choose_balanced_episode_split(
        episode_stats=episode_stats,
        train_ratio=train_ratio,
        num_trials=split_num_trials,
        seed=split_seed,
        verbose=True,
    )

    print(f'Train episode count: {len(train_indices)}')
    print(f'Val episode count: {len(val_indices)}')
    print(f'[INFO] Augmentation (train): photometric_aug={photometric_aug}, spatial_aug={spatial_aug}')
    print('[INFO] Augmentation (val): photometric_aug=False, spatial_aug=False')

    # obtain normalization stats for qpos and action
    norm_stats = get_pose_norm_stats(episode_paths)

    # construct dataset and dataloader
    train_dataset = EpisodicPoseDataset(
        [episode_paths[index] for index in train_indices],
        camera_names,
        chunk_size,
        norm_stats,
        photometric_aug=photometric_aug,
        spatial_aug=spatial_aug,
        image_size=image_size,
        event_channel_indices=event_channel_indices,
    )
    val_dataset = EpisodicPoseDataset(
        [episode_paths[index] for index in val_indices],
        camera_names,
        chunk_size,
        norm_stats,
        photometric_aug=False,
        spatial_aug=False,
        image_size=image_size,
        event_channel_indices=event_channel_indices,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
