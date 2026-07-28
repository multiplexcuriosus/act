import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from transform import TF_mat
from sensor_msgs.msg import PointCloud2, PointField

def get_episode_time_sim(bag):
    time_start = []
    time_pause = []
    time_end = []
    for topic, msg, t in bag.read_messages(topics=['/tocabi/srmt/trajectory']):
        time_start.append(t.to_sec())
    for topic, msg, t in bag.read_messages(topics=['/mujoco_ros_interface/hand_open']):
        if msg.data == 0:
            time_end.append(t.to_sec())
    
    return time_start, time_end

def get_episode_time(bag):
    time_start = []
    time_pause = []
    time_end = []
    for topic, msg, t in bag.read_messages(topics=['/tocabi/guilog']):
        if msg.data == "Upperbody Mode is Changed to #10 (3D Mouse Mode)":
            time_start.append(t.to_sec())
        elif msg.data == "Robot is Freezed!":
            time_pause.append(t.to_sec())
        elif msg.data == "Ready Pose is On!":
            time_end.append(t.to_sec())
    
    return time_start, time_pause, time_end

def im_msg_2_cv_img(img_msg, dir=None, rotate=False):
    image_array = np.frombuffer(img_msg.data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)
    if dir != None:
        stamp = img_msg.header.stamp
        filename = f'{img_msg.header.seq}_{stamp.secs}_{stamp.nsecs:09}.{img_msg.format}'
        cv2.imwrite(os.path.join(dir, filename), image)
    # image = image[:,160:1760]
    # image = cv2.resize(image, (800,600), interpolation=cv2.INTER_AREA)
    # image = image[480:,480:1440]
    # image = cv2.resize(image, (640,480), interpolation=cv2.INTER_AREA)

    return image

def numpy_to_pc2_rgb(header, points):
    assert points.shape[-1] == 6, "Input must be [x, y, z, r, g, b]"
    # Flatten the points array
    points = points.reshape(-1, 6).astype(np.float32)

    # unnormalize RGB values if necessary
    if points[:, 3:].max() <= 1.0:
        points[:, 3:] = points[:, 3:] * 255.0

    # pack RGB into single float32 value
    rgb_uint32 = (
        (points[:, 3].astype(np.uint32) << 16) |
        (points[:, 4].astype(np.uint32) << 8) |
        (points[:, 5].astype(np.uint32))
    )
    rgb_float = rgb_uint32.view(np.float32)

    points = np.hstack((points[:, :3], rgb_float.reshape(-1, 1)))

    # Define the fields of the PointCloud2 message
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]

    # Create the PointCloud2 message
    pc2_msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        fields=fields,
        is_bigendian=False,
        point_step=16,  # 4 bytes per field * 4 fields
        row_step=16 * points.shape[0],
        data=points.tobytes(),
        is_dense=True
    )

    return pc2_msg


def pc2_to_numpy(msg):
    """
    Converts a PointCloud2 message to an Nx3 numpy array (x, y, z).
    
    Parameters:
        msg (sensor_msgs.msg.PointCloud2): The PointCloud2 message.
        
    Returns:
        np.array: Nx3 array where each row contains x, y, z (float64).
    """
    # Get raw data buffer from the PointCloud2 message
    dtype_list = {'names': [], 'formats': [], 'offsets': [],
                    'itemsize': msg.point_step}
    for field in msg.fields:
        dtype_list['names'].append(field.name)
        dtype_list['formats'].append(np.uint32 if field.name == 'rgb' else np.float32)
        dtype_list['offsets'].append(field.offset)
    
    dtype = np.dtype(dtype_list)
    data = np.frombuffer(msg.data, dtype=dtype)

    # Extract x, y, z fields
    xyz = np.vstack((data["x"], data["y"], data["z"])).T
    # Extract rgb fields
    if 'rgb' in data.dtype.names:
        rgb_uint32 = data['rgb'].view(np.uint32)
        r = ((rgb_uint32 >> 16) & 255).astype(np.float32) / 255.0
        g = ((rgb_uint32 >> 8)  & 255).astype(np.float32) / 255.0
        b = (rgb_uint32 & 255).astype(np.float32) / 255.0
        rgb = np.vstack((r, g, b)).T
        xyz = np.hstack((xyz, rgb))

    return xyz

def get_pc_msgs(bag, pc_topic, t_start=None, t_end=None):
    time = []
    pc_msg = []
    for topic, msg, t in bag.read_messages(topics=[pc_topic]):
        cur_time = t.to_sec()
        if t_start is not None and t_end is not None:
            if cur_time < t_start:
                continue
            if cur_time > t_end:
                break
        time.append(cur_time)
        pc_msg.append(msg)

    return time, pc_msg

def transform_points(points, T): # points: Nx3, T: TransformStamped
    translation = np.array([T.transform.translation.x, T.transform.translation.y, T.transform.translation.z]).reshape(3, 1)
    orientation = [T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
    rot_mat = R.from_quat(orientation).as_matrix()
    
    transformed = (rot_mat @ points.T) + translation
    return transformed.T

def transform_points(points, translation, quaternion): # points: Nx3
    translation = np.array(translation).reshape(3, 1)
    rot_mat = R.from_quat(quaternion).as_matrix()
    
    transformed = (rot_mat @ points.T) + translation
    return transformed.T

def rgb_to_hsv(rgb):
    """
    Convert RGB values to HSV.
    
    Parameters:
        rgb (np.array): Nx3 array of RGB values.
        
    Returns:
        np.array: Nx3 array of HSV values.
    """
    hsv = np.zeros_like(rgb)
    
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    delta = maxc - minc
    
    hsv[:, 2] = maxc  # Value channel
    
    mask = delta > 0
    hsv[mask, 1] = delta[mask] / maxc[mask]  # Saturation channel
    
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    hsv[mask, 0] = np.where(
        r[mask] == maxc[mask], (g[mask] - b[mask]) / delta[mask],
        np.where(g[mask] == maxc[mask], 2.0 + (b[mask] - r[mask]) / delta[mask], 4.0 + (r[mask] - g[mask]) / delta[mask])
    )
    
    hsv[:, 0] = (hsv[:, 0] / 6.0) % 1.0  # Normalize hue to [0, 1]
    
    return hsv
    
class ColorPicker:
    def __init__(self, img_msg):
        self.bridge = CvBridge()
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        self.hsv = None
        self.clicked = False

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        while not self.clicked:
            cv2.imshow("Image", self.cv_image)
            if cv2.waitKey(10) & 0xFF == 27:  # exit with ESC
                break
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.cv_image is not None:
            b, g, r = self.cv_image[y, x]
            self.hsv = rgb_to_hsv(np.array([[r, g, b]]) / 255.0)
            print(f"Clicked pixel (x={x}, y={y}) -> RGB=({r}, {g}, {b}), HSV={self.hsv[0]}")
            self.clicked = True


joint_names = ["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint",
               "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
               "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint",
               "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
               "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
               "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Armlink_Joint",
               "L_Elbow_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
               "Neck_Joint", "Head_Joint",
               "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
               "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"]
joint_limits = [[-0.6, 0.6], [-3, 3], [-1.5, 3], [-0.5, 3], [-1.4, 1], [-0.664, 0.664],
                [-0.6, 0.6], [-3, 3], [-1.5, 3], [-0.5, 3], [-1.4, 1], [-0.664, 0.664],
                [-3, 3], [-3, 3], [-3, 3],
                [-2.09, 1.54], [-3.15, 3.15], [-1.92, 1.92], [-3.15, 3.15],
                [-2.8, -0.1], [-3.15, 3.15], [-1.57, 1.57], [-2.094, 2.094],
                [-1, 1], [-1, 1],
                [-1.54, 2.09], [-3.15, 3.15], [-1.92, 1.92], [-3.15, 3.15],
                [0.1, 2.8], [-3.15, 3.15], [-1.57, 1.57], [-2.094, 2.094]]

# for full joint
# joi = np.arange(33)
# nrow, ncol = 8, 5
# plot_pos = [0, 1, 2, 3, 4, 5,
#               8, 9, 10, 11, 12, 13,
#               16, 17, 18,
#               24, 25, 26, 27, 28, 29, 30, 31,
#               19, 20,
#               32, 33, 34, 35, 36, 37, 38, 39]
# for right arm + head
# joi = [12, 13, 14, 23, 24, 
#        25, 26, 27, 28, 
#        29, 30, 31, 32]
# nrow, ncol = 5, 3
# plot_pos = [0, 1, 2, 3, 4,
#               5, 6, 7, 8,
#               10, 11, 12, 13]
# for upper body
joi = np.arange(12, 33)
nrow, ncol = 8, 3
plot_pos = [12, 11, 10,
            0, 1, 2, 3, 4, 5, 6, 7,
            9, 8,
            16, 17, 18, 19, 20, 21, 22, 23]

def plot_jointstates(time, position, fig=None, axs=None, show=True):#, color='b'):
    if not fig:
        fig, axs = plt.subplots(nrow, ncol)
    
    for p, j in zip(plot_pos, joi):
        r = p%nrow
        c = p//nrow
        axs[r][c].set_title(joint_names[j], {'fontsize': 8})
        axs[r][c].plot(time, position[:,j])#, color=color)
        # axs[r][c].set_ylim(joint_limits[j])
        axs[r][c].set_xlim([400, 440])
        axs[r][c].set_ylabel('angle(rad)', {'fontsize': 8})
        axs[r][c].set_xlabel('time(sec)', {'fontsize': 8})

    if show:
        plt.subplots_adjust(hspace=2.0, wspace=0.3)
        fig.show()
    
    return fig, axs

def plot_handpose(pose):
    hand_pos_x = [p[1] for p in pose]
    hand_pos_y = [p[2] for p in pose]
    hand_pos_z = [p[3] for p in pose]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(hand_pos_x, hand_pos_y, hand_pos_z)
    ax.axis('equal')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_zlabel('z(m)')
    fig.show()

def save_video(video, fps, video_path=None):
    h, w, _ = video[0].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for image in video:
        out.write(image)
    out.release()
    # print(f'Saved video to: {video_path}')

def lpf(input, prev_res, sampling_freq, cutoff_freq):
    rc = 1 / (cutoff_freq * 2 * np.pi)
    dt = 1 / sampling_freq
    a = dt / (rc + dt)

    return prev_res + a * (input - prev_res)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
