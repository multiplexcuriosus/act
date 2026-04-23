import os
import numpy as np
import cv2
import rosbag
from tqdm import tqdm
import h5py
import utils
from transform import TF_mat
import gc
import matplotlib.pyplot as plt
import csv


def get_next_available_episode_idx(dataset_dir, start_idx=0):
    idx = start_idx
    while os.path.exists(os.path.join(dataset_dir, f'episode_{idx}.hdf5')):
        idx += 1
    return idx


bagfile_dir = '/media/lyh/SSD2TB/rosbag/20250723'
bagfile_list = [bagfile for bagfile in os.listdir(bagfile_dir) if bagfile.endswith('.bag')]
bagfile_list.sort()

task_name = 'pick_n_place_small'

dataset_dir = f'/media/lyh/SSD2TB/act/data/real_tocabi_{task_name}'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
camera_names = ['left', 'right']
FPS = 30

episode_len = []
episode_idx = get_next_available_episode_idx(dataset_dir, start_idx=0)
for bagfile in bagfile_list:
    if os.path.exists(os.path.join(bagfile_dir, f'episode_timestamp_{bagfile[:-4]}.txt')):
        with open(os.path.join(bagfile_dir, f'episode_timestamp_{bagfile[:-4]}.txt'), 'r') as f:
            reader = csv.reader(f)
            episode_time = list(reader)
            t0 = float(episode_time[0][0])
            episode_time = np.array(episode_time[1:], dtype=float)
            time_start = episode_time[:,0] + t0
            time_pause = episode_time[:,1] + t0
            time_end = episode_time[:,2] + t0
            num_episode = len(time_start)
            print(f'[INFO] {bagfile} has {len(time_start)} episodes')
    else:
        with rosbag.Bag(os.path.join(bagfile_dir, bagfile)) as bag:
            time_start, time_pause, time_end = utils.get_episode_time(bag)
        # handle exception
        if len(time_start) != len(time_pause) or len(time_start) != len(time_end):
            print(f'[WARNING] {bagfile} has different start, pause and end time')
            print( '          handling exception...')
            time_start_new = []
            time_pause_new = []
            time_end_new = []
            pause_idx = 0
            end_idx = 0
            last_time_end = 0
            for t_start in time_start:
                if t_start < last_time_end:
                    continue
                if pause_idx >= len(time_pause) or end_idx >= len(time_end):
                    break
                while t_start > time_pause[pause_idx]:
                    pause_idx += 1
                while t_start > time_end[end_idx]:
                    end_idx += 1
                time_start_new.append(t_start)
                if time_pause[pause_idx] > time_end[end_idx]:
                    time_pause_new.append(time_end[end_idx]-1/FPS)
                    time_end_new.append(time_end[end_idx])
                    end_idx += 1
                else:
                    time_pause_new.append(time_pause[pause_idx])
                    time_end_new.append(time_end[end_idx])
                    pause_idx += 1
                    end_idx += 1
                last_time_end = time_end_new[-1]
            time_start = time_start_new
            time_pause = time_pause_new
            time_end = time_end_new
        if len(time_start) != len(time_pause) or len(time_start) != len(time_end):
            print(f'[ ERROR ] failed to handle exception')
            print(len(time_start), len(time_pause), len(time_end))
            print(time_start)
            print(time_pause)
            print(time_end)
            continue
        else:
            num_episode = len(time_start)
            print(f'[INFO] {bagfile} has {len(time_start)} episodes')
            with open(os.path.join(bagfile_dir, f'episode_timestamp_{bagfile[:-4]}.txt'), 'w') as f:
                t0 = time_start[0]
                f.write(f'{t0:.6f}\n')
                for t1, t2, t3 in zip(time_start, time_pause, time_end):
                    f.write(f'{t1-t0:.6f}, {t2-t0:.6f}, {t3-t0:.6f}\n')

    with rosbag.Bag(os.path.join(bagfile_dir, bagfile)) as bag:
        data = {'joint': [], 'joint_time': [],
                'pose': [], 'pose_time': [],
                'hand_open_time': [], 'hand_close_time': [],
                'img_msg_left': [], 'img_left_time': [],
                'img_msg_right': [], 'img_right_time': []}
        time_idx = 0
        iter = tqdm(range(num_episode))
        
        for topic, msg, t in bag.read_messages(topics=[
                '/tocabi/jointstates',
                '/tocabi/robot_poses',
                '/tocabi_hand/on',
                '/cam_LEFT/image_raw/compressed',
                '/cam_RIGHT/image_raw/compressed']):
            cur_time = t.to_sec()
            if cur_time < time_start[time_idx]:
                continue
            if cur_time < time_end[time_idx] + 4.0:
                if topic == '/tocabi/jointstates':
                    data['joint_time'].append(cur_time)
                    data['joint'].append(msg.position)
                elif topic == '/tocabi/robot_poses':
                    data['pose_time'].append(cur_time)
                    data['pose'].append(TF_mat.from_posearray_msg(msg).as_matrix())
                elif topic == '/tocabi_hand/on':
                    if msg.data:
                        data['hand_close_time'].append(cur_time)
                    else:
                        data['hand_open_time'].append(cur_time)
                elif topic == '/cam_LEFT/image_raw/compressed':
                    data['img_left_time'].append(cur_time)
                    data['img_msg_left'].append(msg)
                elif topic == '/cam_RIGHT/image_raw/compressed':
                    data['img_right_time'].append(cur_time)
                    data['img_msg_right'].append(msg)
            else:   # end of episode
                rotbot_TF = np.array(data['pose'])

                # variables for time synchronization
                joint_idx = 0
                robot_pose_idx = 0
                hand_close_idx = 0
                hand_open_idx = 0
                img_left_idx = 0
                img_right_idx = 0

                joint = []
                robot_poses_indices = []
                hand_action = []
                image_left = []
                image_right = []

                # if data['hand_close_time'][0] < data['hand_open_time'][0]:
                #     cur_hand_action = 0
                # else:
                #     cur_hand_action = 1
                cur_hand_action = 0
                # make time synchronized data
                timesteps = np.concatenate([np.arange(time_start[time_idx], time_pause[time_idx] + 1/FPS, 1/FPS), np.arange(time_end[time_idx], time_end[time_idx]+4, 1/FPS)])
                for t in timesteps:
                    while data['joint_time'][joint_idx] < t and joint_idx < len(data['joint_time'])-1:
                        joint_idx += 1
                    while data['pose_time'][robot_pose_idx] < t and robot_pose_idx < len(data['pose_time'])-1:
                        robot_pose_idx += 1
                    while data['img_left_time'][img_left_idx] < t and img_left_idx < len(data['img_left_time'])-1:
                        img_left_idx += 1
                    while data['img_right_time'][img_right_idx] < t and img_right_idx < len(data['img_right_time'])-1:
                        img_right_idx += 1
                    
                    joint.append(data['joint'][joint_idx])
                    robot_poses_indices.append(robot_pose_idx)

                    if cur_hand_action == 0 and hand_close_idx < len(data['hand_close_time']):
                        if data['hand_close_time'][hand_close_idx] < t:
                            cur_hand_action = 1
                            hand_close_idx += 1
                    elif hand_open_idx < len(data['hand_open_time']):
                        if data['hand_open_time'][hand_open_idx] < t:
                            cur_hand_action = 0
                            hand_open_idx += 1
                    hand_action.append(cur_hand_action)

                    img_left = utils.im_msg_2_cv_img(data['img_msg_left'][img_left_idx])
                    img_right = utils.im_msg_2_cv_img(data['img_msg_right'][img_right_idx])#, rotate=True)
                    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                    img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                    image_left.append(np.array(img_left_rgb, dtype=np.uint8))
                    image_right.append(np.array(img_right_rgb, dtype=np.uint8))
                
                pelvis_pose = TF_mat(rotbot_TF[robot_poses_indices, 0])
                head_pose = TF_mat(rotbot_TF[robot_poses_indices, 2])
                rhand_pose = TF_mat(rotbot_TF[robot_poses_indices, 3])
                hand_action = np.array(hand_action).reshape([-1, 1])
                
                data_len = len(timesteps)
                episode_len.append(data_len)

                # convert TF matrix of rhand to 12D vector in two frame (pelvis, head)
                head_pelvis = TF_mat.mul(pelvis_pose.inverse(), head_pose)
                rhand_pelvis = TF_mat.mul(pelvis_pose.inverse(), rhand_pose)
                rhand_head = TF_mat.mul(head_pose.inverse(), rhand_pose)
                ee_pose_global = rhand_pelvis.as_matrix()[:,:3,:].swapaxes(-1, -2).reshape([-1, 12])
                ee_pose_rel = rhand_head.as_matrix()[:,:3,:].swapaxes(-1, -2).reshape([-1, 12])

                # define action as state after k timesteps
                k = 3
                joint_action = np.concatenate([joint[k:], np.repeat(joint[-1:], k, axis=0)], axis=0)
                ee_action_global = np.concatenate([ee_pose_global[k:], np.repeat(ee_pose_global[-1:], k, axis=0)], axis=0)
                ee_action_rel = np.concatenate([ee_pose_rel[k:], np.repeat(ee_pose_rel[-1:], k, axis=0)], axis=0)
                # for hand, define state from action
                hand_state = np.concatenate([hand_action[:1], hand_action[:-1]], axis=0)
                k = 15
                hand_action = np.concatenate([hand_state[k:], np.repeat(hand_state[-1:], k, axis=0)], axis=0)

                # append hand state & action to each state
                joint = np.concatenate([joint, hand_state], axis=1)
                ee_pose_global = np.concatenate([ee_pose_global, hand_state], axis=1)
                ee_pose_rel = np.concatenate([ee_pose_rel, hand_state], axis=1)
                joint_action = np.concatenate([joint_action, hand_action], axis=1)
                ee_action_global = np.concatenate([ee_action_global, hand_action], axis=1)
                ee_action_rel = np.concatenate([ee_action_rel, hand_action], axis=1)

                data_dict = {
                    '/observations/qpos': joint,
                    '/observations/ee_pose_global': ee_pose_global,
                    '/observations/ee_pose_rel': ee_pose_rel,
                    '/action': joint_action,
                    '/ee_action_global': ee_action_global,
                    '/ee_action_rel': ee_action_rel,
                    '/observations/images/left': image_left,
                    '/observations/images/right': image_right,
                    '/observations/head_pose': head_pelvis.as_matrix(),
                    '/observations/rhand_pose': rhand_pelvis.as_matrix(),
                }

                dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
                with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                    root.attrs['sim'] = True
                    obs = root.create_group('observations')
                    image = obs.create_group('images')
                    for cam_name in camera_names:
                        _ = image.create_dataset(cam_name, (data_len, 600, 800, 3), dtype='uint8',
                                                    chunks=(1, 600, 800, 3),)
                                                    # compression='gzip',compression_opts=2,)
                                                    # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                    qpos = obs.create_dataset('qpos', (data_len, 34))
                    ee_pose_global = obs.create_dataset('ee_pose_global', (data_len, 13))
                    ee_pose_rel = obs.create_dataset('ee_pose_rel', (data_len, 13))
                    head_pose = obs.create_dataset('head_pose', (data_len, 4, 4))
                    rhand_pose = obs.create_dataset('rhand_pose', (data_len, 4, 4))
                    action = root.create_dataset('action', (data_len, 34))
                    ee_action_global = root.create_dataset('ee_action_global', (data_len, 13))
                    ee_action_rel = root.create_dataset('ee_action_rel', (data_len, 13))

                    for name, array in data_dict.items():
                        root[name][...] = array
                
                episode_idx = get_next_available_episode_idx(dataset_dir, start_idx=episode_idx + 1)
                time_idx += 1

                del data, image_left, image_right, data_dict
                gc.collect()

                iter.update()

                if time_idx == num_episode:
                    break

                data = {'joint': [], 'joint_time': [],
                        'pose': [], 'pose_time': [],
                        'hand_open_time': [], 'hand_close_time': [],
                        'img_msg_left': [], 'img_left_time': [],
                        'img_msg_right': [], 'img_right_time': []}

print(max(episode_len))
plt.plot(episode_len)
plt.title('Episode length')
plt.xlabel('Episode index')
plt.ylabel('Episode length')
plt.savefig(os.path.join(bagfile_dir, 'episode_length.svg'))
