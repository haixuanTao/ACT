import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = 'data/'

# checkpoint directory
CHECKPOINT_DIR = 'checkpoints/'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

# robot port names
ROBOT_PORTS = {
    'leader': '/dev/ttyDXL_master_right',
    'follower': '/dev/ttyDXL_puppet_right'
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 250,
    'state_dim': 7,
    'action_dim': 7,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['cam_left_wrist', 'cam_right_wrist', 'cam_low', 'cam_high'],
    'camera_port': ['/dev/video2', '/dev/video22', '/dev/video14', '/dev/video8'],
    #'camera_names': ['cam_right_wrist'],
    #'camera_port': ['/dev/video22'],
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['cam_left_wrist', 'cam_right_wrist', 'cam_low', 'cam_high'],
    #'camera_names': ['cam_right_wrist'],
    'policy_class': 'ACT',
    'temporal_agg': False
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 2000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}