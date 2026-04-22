import pathlib

### Task parameters
DATA_DIR = './data/act/data'
# DATA_DIR = '/media/embodied_ai/SSD2TB/act/data'
SIM_TASK_CONFIGS = {
    'toy_circle_chase': {
    'dataset_dir': '../toy_il/data/out',
    'num_episodes': 10,
    'episode_len': 240,
    'camera_names': ['main'],
    'model_dof': 2,
    },
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 5,
        'episode_len': 400,
        'camera_names': ['top'],
        'model_dof': 14
    },

    'sim_panda_catch_ball':{
        'dataset_dir': '/external/data/maniskill/PickBall',
        'num_episodes': 200,
        'episode_len': 500,
        'camera_names': ['wrist'],
        'model_dof': 7
    },
    
    'real_tocabi_pick_n_place':{
        'dataset_dir': DATA_DIR + '/real_tocabi_pick_n_place',
        'num_episodes': 200,
        'episode_len': 1000,
        'camera_names': ['stereo'],
        'model_dof': 9
    },

    'real_panda_pick_n_place':{
        'dataset_dir': DATA_DIR + '/real_panda_pick_n_place',
        'num_episodes': 100,
        'episode_len': 650,
        'camera_names': ['left', 'hand', 'right'],
        'model_dof': 8
    },

    'real_panda_peg_in_hole':{
        'dataset_dir': DATA_DIR + '/real_panda_peg_in_hole',
        'num_episodes': 48,
        'episode_len': 600,
        'camera_names': ['top'],
        'model_dof': 7
    },

    'real_tocabi_task_open':{
        'dataset_dir': DATA_DIR + '/real_tocabi_open_task_space',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['left'],
        'model_dof': 13
    },

    'real_tocabi_open_head':{
        'dataset_dir': DATA_DIR + '/real_tocabi_open_head_only',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['left'],
        'model_dof': 5
    },

    'real_tocabi_open':{
        'dataset_dir': DATA_DIR + '/real_tocabi_open_cropped_smoothed_action',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['left', 'right'],
        'model_dof': 14
    },

    'real_tocabi_pick_cup':{
        'dataset_dir': DATA_DIR + '/real_tocabi_pick_cup_cropped_smoothed_action',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['left', 'right'],
        'model_dof': 14
    },

    'real_tocabi_insert':{
        'dataset_dir': DATA_DIR + '/real_tocabi_insert_cropped_smoothed_action',
        'num_episodes': 50,
        'episode_len': 450,
        'camera_names': ['left', 'right'],
        'model_dof': 14
    },

    'real_tocabi_pick':{
        'dataset_dir': DATA_DIR + '/real_tocabi_pick_from_ready_pose',
        'num_episodes': 32,
        'episode_len': 450,
        'camera_names': ['left'],
        'model_dof': 13
    },

    'real_tocabi_pickup':{
        'dataset_dir': DATA_DIR + '/real_tocabi_pickup_augmented_cropped',
        'num_episodes': 65,
        'episode_len': 600,
        'camera_names': ['left'],
        'model_dof': 14
    },

    'real_tocabi_place':{
        'dataset_dir': DATA_DIR + '/real_tocabi_place',
        'num_episodes': 12,
        'episode_len': 900,
        'camera_names': ['left', 'right'],
        'model_dof': 14
    },

    'real_tocabi_approach_tabletop':{
        'dataset_dir': DATA_DIR + '/real_tocabi_approach_tabletop',
        'num_episodes': 22,
        'episode_len': 450,
        'camera_names': ['head'],
        'model_dof': 11
    },

    'sim_tocabi_approach_mustard':{
        'dataset_dir': DATA_DIR + '/sim_tocabi_approach_mustard',
        'num_episodes': 20,
        'episode_len': 223,
        'camera_names': ['top'],
        'model_dof': 8
    },

    'sim_tocabi_approach_tabletop':{
        'dataset_dir': DATA_DIR + '/sim_tocabi_approach_tabletop',
        'num_episodes': 45,
        'episode_len': 250,
        'camera_names': ['top'],
        'model_dof': 8
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top'],
        'model_dof': 14
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top'],
        'model_dof': 14
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top'],
        'model_dof': 14
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
