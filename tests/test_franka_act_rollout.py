import importlib.util
import os
import sys
import types
import unittest

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROLLOUT_PATH = os.path.join(REPO_ROOT, "franka_act_rollout.py")


def _install_rollout_stubs() -> None:
    if "cv_bridge" not in sys.modules:
        cv_bridge = types.ModuleType("cv_bridge")

        class _CvBridge:
            def imgmsg_to_cv2(self, msg, desired_encoding="rgb8"):
                return msg.array

        cv_bridge.CvBridge = _CvBridge
        sys.modules["cv_bridge"] = cv_bridge

    if "sensor_msgs" not in sys.modules:
        sensor_msgs = types.ModuleType("sensor_msgs")
        sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

        class _Image:
            pass

        class _JointState:
            pass

        sensor_msgs_msg.Image = _Image
        sensor_msgs_msg.JointState = _JointState
        sensor_msgs.msg = sensor_msgs_msg
        sys.modules["sensor_msgs"] = sensor_msgs
        sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    if "geometry_msgs" not in sys.modules:
        geometry_msgs = types.ModuleType("geometry_msgs")
        geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

        class _Vector3:
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0

        class _Twist:
            def __init__(self):
                self.linear = _Vector3()
                self.angular = _Vector3()

        class _Header:
            def __init__(self):
                self.stamp = None
                self.frame_id = ""

        class _TwistStamped:
            def __init__(self):
                self.header = _Header()
                self.twist = _Twist()

        geometry_msgs_msg.TwistStamped = _TwistStamped
        geometry_msgs.msg = geometry_msgs_msg
        sys.modules["geometry_msgs"] = geometry_msgs
        sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

    if "std_msgs" not in sys.modules:
        std_msgs = types.ModuleType("std_msgs")
        std_msgs_msg = types.ModuleType("std_msgs.msg")

        class _Bool:
            def __init__(self):
                self.data = False

        class _Float32MultiArray:
            def __init__(self):
                self.data = []

        std_msgs_msg.Bool = _Bool
        std_msgs_msg.Float32MultiArray = _Float32MultiArray
        std_msgs.msg = std_msgs_msg
        sys.modules["std_msgs"] = std_msgs
        sys.modules["std_msgs.msg"] = std_msgs_msg

    if "rclpy" not in sys.modules:
        rclpy = types.ModuleType("rclpy")
        rclpy_node = types.ModuleType("rclpy.node")
        rclpy_qos = types.ModuleType("rclpy.qos")

        class _Node:
            pass

        rclpy_node.Node = _Node
        rclpy_qos.qos_profile_sensor_data = object()
        rclpy.node = rclpy_node
        rclpy.qos = rclpy_qos
        sys.modules["rclpy"] = rclpy
        sys.modules["rclpy.node"] = rclpy_node
        sys.modules["rclpy.qos"] = rclpy_qos

    if "policy" not in sys.modules:
        policy = types.ModuleType("policy")

        class _ACTPolicy:
            def __init__(self, *_args, **_kwargs):
                pass

        policy.ACTPolicy = _ACTPolicy
        policy._build_image_normalizer = lambda image_channels: image_channels
        policy.transforms = types.SimpleNamespace(
            Normalize=lambda mean, std: {"mean": mean, "std": std}
        )
        sys.modules["policy"] = policy


def _load_rollout_module():
    _install_rollout_stubs()
    module_name = "franka_act_rollout_test_module"
    spec = importlib.util.spec_from_file_location(module_name, ROLLOUT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {ROLLOUT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _FakeGate:
    def __init__(self, sequence):
        self.sequence = sequence

    def claim_latest_ready_sequence(self, _buffer_len):
        value = self.sequence
        self.sequence = None
        return value


class FrankaActRolloutHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rollout = _load_rollout_module()

    def test_build_qpos_state_dim_7(self):
        joint_names = self.rollout.ARM_JOINT_NAMES + self.rollout.FINGER_JOINT_NAMES
        joint_pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.1, 0.2], dtype=np.float32)
        qpos = self.rollout.build_qpos_from_joint_state(joint_names, joint_pos, state_dim=7)
        np.testing.assert_allclose(qpos, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32))

    def test_build_qpos_state_dim_8(self):
        joint_names = self.rollout.ARM_JOINT_NAMES + self.rollout.FINGER_JOINT_NAMES
        joint_pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.1, 0.2], dtype=np.float32)
        qpos = self.rollout.build_qpos_from_joint_state(joint_names, joint_pos, state_dim=8)
        np.testing.assert_allclose(
            qpos,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.3], dtype=np.float32),
        )

    def test_preprocess_rgb_history_shapes_and_channel_order(self):
        for history_len in (1, 2, 3):
            frames = [np.full((2, 2, 3), fill_value=value, dtype=np.uint8) for value in (10, 20, 30)[:history_len]]
            tensor = self.rollout.preprocess_rgb_history_images(frames, image_size=2)
            self.assertEqual(tuple(tensor.shape), (1, 1, 3 * history_len, 2, 2))
            np_tensor = tensor.numpy()
            for idx, value in enumerate((10, 20, 30)[:history_len]):
                channel_slice = np_tensor[0, 0, idx * 3:(idx + 1) * 3]
                self.assertTrue(np.allclose(channel_slice, value / 255.0))

    def test_image_sequence_gate_requires_full_history_and_prevents_duplicates(self):
        gate = self.rollout.ImageSequenceGate(required_history=2)
        gate.note_image_received()
        self.assertIsNone(gate.claim_latest_ready_sequence(buffer_len=1))
        gate.note_image_received()
        self.assertEqual(gate.claim_latest_ready_sequence(buffer_len=2), 2)
        self.assertIsNone(gate.claim_latest_ready_sequence(buffer_len=2))
        gate.note_image_received()
        self.assertEqual(gate.claim_latest_ready_sequence(buffer_len=2), 3)

    def test_resolve_rollout_configuration_defaults_and_history_channels(self):
        base_stats = {
            "qpos_mean": np.zeros(7, dtype=np.float32),
            "qpos_std": np.ones(7, dtype=np.float32),
            "action_mean": np.array([0.0, 0.0], dtype=np.float32),
            "action_std": np.array([1.0, 1.0], dtype=np.float32),
        }
        args = types.SimpleNamespace(
            state_dim=None,
            action_dim=None,
            camera_name="rgb",
            rollout_mode="intercept",
            temporal_agg=False,
            use_bce_last_action_dim=True,
        )

        cfg = self.rollout.resolve_rollout_configuration(args, dict(base_stats))
        self.assertEqual(cfg["rgb_history_frames"], 1)
        self.assertEqual(cfg["image_channels"], 3)

        stats_h2 = dict(base_stats)
        stats_h2.update({
            "rgb_history_frames": 2,
            "image_channels": 6,
            "camera_names": ["rgb"],
            "rgb_frame_order": "oldest_to_newest",
        })
        cfg_h2 = self.rollout.resolve_rollout_configuration(args, stats_h2)
        self.assertEqual(cfg_h2["rgb_history_frames"], 2)
        self.assertEqual(cfg_h2["image_channels"], 6)

        stats_h3 = dict(base_stats)
        stats_h3.update({
            "rgb_history_frames": 3,
            "image_channels": 9,
            "camera_names": ["rgb"],
            "rgb_frame_order": "oldest_to_newest",
        })
        cfg_h3 = self.rollout.resolve_rollout_configuration(args, stats_h3)
        self.assertEqual(cfg_h3["rgb_history_frames"], 3)
        self.assertEqual(cfg_h3["image_channels"], 9)

        bad_stats = dict(base_stats)
        bad_stats.update({
            "rgb_history_frames": 2,
            "image_channels": 7,
            "camera_names": ["rgb"],
            "rgb_frame_order": "oldest_to_newest",
        })
        with self.assertRaisesRegex(ValueError, "image_channels == 3 \* rgb_history_frames"):
            self.rollout.resolve_rollout_configuration(args, bad_stats)

    def test_convert_intercept_action_uses_only_first_dim_for_denorm_and_sigmoid_for_bce(self):
        raw_action = np.array([0.5, 0.0], dtype=np.float32)
        action_mean = np.array([0.2, 0.0], dtype=np.float32)
        action_std = np.array([0.4, 1.0], dtype=np.float32)
        target_s_m, execute_logit, execute_probability = self.rollout.convert_intercept_action(
            raw_action,
            action_mean,
            action_std,
        )
        self.assertAlmostEqual(target_s_m, 0.4)
        self.assertAlmostEqual(execute_logit, 0.0)
        self.assertAlmostEqual(execute_probability, 0.5)

        zero_target, _, _ = self.rollout.convert_intercept_action(
            np.array([0.0, -2.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(zero_target, 0.0)

    def test_intercept_publication_emits_exact_pair(self):
        node = object.__new__(self.rollout.FrankaActRolloutNode)
        node.action_mean = np.array([1.0, 0.0], dtype=np.float32)
        node.action_std = np.array([2.0, 1.0], dtype=np.float32)
        node.prediction_pub = _FakePublisher()

        target_s_m, execute_logit, execute_probability = node.publish_intercept_prediction(
            np.array([0.25, 0.0], dtype=np.float32)
        )

        self.assertAlmostEqual(target_s_m, 1.5)
        self.assertAlmostEqual(execute_logit, 0.0)
        self.assertAlmostEqual(execute_probability, 0.5)
        self.assertEqual(len(node.prediction_pub.messages), 1)
        self.assertEqual(node.prediction_pub.messages[0].data, [1.5, 0.5])

    def test_select_current_intercept_action_uses_only_first_query(self):
        action_chunk = torch.tensor(
            [[[1.0, 2.0], [9.0, 9.0], [8.0, 8.0]]],
            dtype=torch.float32,
        )
        raw_action = self.rollout.select_current_intercept_action(action_chunk, num_queries=3)
        self.assertTrue(torch.equal(raw_action, torch.tensor([1.0, 2.0], dtype=torch.float32)))

    def test_timer_cb_dispatches_intercept_without_legacy_commands(self):
        node = object.__new__(self.rollout.FrankaActRolloutNode)
        calls = []
        node.rollout_mode = "intercept"
        node.ready = lambda: True
        node.running = True
        node.rgb_buffer = [object(), object()]
        node.image_sequence_gate = _FakeGate(sequence=5)
        node.get_logger = lambda: _FakeLogger()
        node.run_intercept_policy_step = lambda image_sequence: calls.append(("intercept", image_sequence))
        node.run_twist_gripper_policy_step = lambda: (_ for _ in ()).throw(AssertionError("legacy path used"))
        node.publish_initial_gripper_open = lambda: (_ for _ in ()).throw(AssertionError("gripper command used"))

        node.timer_cb()
        self.assertEqual(calls, [("intercept", 5)])

    def test_timer_cb_dispatches_legacy_path_in_twist_gripper_mode(self):
        node = object.__new__(self.rollout.FrankaActRolloutNode)
        calls = []
        node.rollout_mode = "twist_gripper"
        node.ready = lambda: True
        node.running = True
        node.sent_initial_gripper_state = False
        node.get_logger = lambda: _FakeLogger()
        node.publish_initial_gripper_open = lambda: calls.append("initial_open")
        node.run_intercept_policy_step = lambda _image_sequence: (_ for _ in ()).throw(AssertionError("intercept path used"))
        node.image_sequence_gate = _FakeGate(sequence=7)
        node.rgb_buffer = [object()]
        node.run_twist_gripper_policy_step = lambda: calls.append("legacy_step")

        node.timer_cb()
        self.assertEqual(calls, ["initial_open", "legacy_step"])


if __name__ == "__main__":
    unittest.main()