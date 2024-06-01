from config.config import TASK_CONFIG, ROBOT_PORTS
import os
import cv2
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from time import sleep, time
from training.utils import pwm2pos, pwm2vel

from robot import Robot

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='sort4')
parser.add_argument('--num_episodes', type=int, default=15)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

cfg = TASK_CONFIG


def capture_image(cam):
    # Capture a single frame
    _, frame = cam.read()
    # Generate a unique filename with the current date and time
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # Define your crop coordinates (top left corner and bottom right corner)
    # x1, y1 = 400, 0  # Example starting coordinates (top left of the crop rectangle)
    # x2, y2 = 1600, 900  # Example ending coordinates (bottom right of the crop rectangle)
    # # Crop the image
    # image = image[y1:y2, x1:x2]
    # Resize the image
    image = cv2.resize(image, (cfg['cam_width'], cfg['cam_height']), interpolation=cv2.INTER_AREA)

    return image


MAX_MASTER_GRIPER = 2554
MAX_PUPPET_GRIPER = 3145

MIN_MASTER_GRIPER = 1965
MIN_PUPPET_GRIPER = 2500

LEADER_GRIPPER_IDX = 6

FOLLOWER_SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
LEADER_SERVO_IDS = [1, 2, 4, 6, 7, 8, 9]

def adapt_gripper_range_from_leader_to_follower(action):
    gripper = action[LEADER_GRIPPER_IDX]
    gripper = (gripper - MIN_MASTER_GRIPER) * (MAX_PUPPET_GRIPER - MIN_PUPPET_GRIPER)
    gripper /= (MAX_MASTER_GRIPER - MIN_MASTER_GRIPER)
    gripper += MIN_PUPPET_GRIPER
    action[LEADER_GRIPPER_IDX] = gripper
    return action

def add_redondant_servos_from_leader_to_follower(action):
    new_action = np.array([
        action[0],
        action[1],
        action[1],
        action[2],
        action[2],
        action[3],
        action[4],
        action[5],
        action[6],
    ])
    return new_action

def drop_redondant_servos_from_follower(state):
    new_state = np.array([
        state[0],
        state[1],
        state[3],
        state[5],
        state[6],
        state[7],
        state[8],
    ])
    return new_state



if __name__ == "__main__":
    # init camera
    cams = [cv2.VideoCapture(p) for p in cfg['camera_port']]
    # Check if the camera opened successfully
    if not all(c.isOpened() for c in cams):
        raise IOError("Cannot open camera")
    # init follower
    follower = Robot(device_name=ROBOT_PORTS['follower'], servo_ids=FOLLOWER_SERVO_IDS)
    # init leader
    leader = Robot(device_name=ROBOT_PORTS['leader'], servo_ids=LEADER_SERVO_IDS)
    leader.set_trigger_torque()

    
    for i in range(num_episodes):
        # bring the follower to the leader and start camera
        for _ in range(200):
            action = leader.read_position()
            action = adapt_gripper_range_from_leader_to_follower(action)
            follower_action = add_redondant_servos_from_leader_to_follower(action)
            follower.set_goal_pos(follower_action)
            _ = [capture_image(c) for c in cams]
        os.system(f'spd-say "go {i}"')
        # init buffers
        obs_replay = []
        action_replay = []
        for i in tqdm(range(cfg['episode_len'])):
            # observation
            qpos = follower.read_position()
            qvel = follower.read_velocity()

            qpos = drop_redondant_servos_from_follower(qpos)
            qvel = drop_redondant_servos_from_follower(qvel)

            images = [capture_image(c) for c in cams]
            images_stacked = np.hstack(images)
            images_stacked = cv2.cvtColor(images_stacked, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', images_stacked) 
            obs = {
                'qpos': pwm2pos(qpos),
                'qvel': pwm2vel(qvel),
                'images': {cn : im for im, cn in zip(images, cfg['camera_names'])}
            }
            # action (leader's position)
            action = leader.read_position()
            action = adapt_gripper_range_from_leader_to_follower(action)
            follower_action = add_redondant_servos_from_leader_to_follower(action)
            # apply action
            follower.set_goal_pos(follower_action)
            action = pwm2pos(action)
            # store data
            obs_replay.append(obs)
            action_replay.append(action)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        os.system('spd-say "stop"')

        # disable torque
        #leader._disable_torque()
        #follower._disable_torque()

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            # store the images
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

        t0 = time()
        max_timesteps = len(data_dict['/observations/qpos'])
        # create data dir if it doesn't exist
        data_dir = os.path.join(cfg['dataset_dir'], task)
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        # count number of files in the directory
        idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
        dataset_path = os.path.join(data_dir, f'episode_{idx}')
        # save the data
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                try:
                    root[name][...] = array
                except:
                    breakpoint()
    
    leader._disable_torque()
    follower._disable_torque()
