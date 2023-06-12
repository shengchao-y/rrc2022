"""Example policy for Real Robot Challenge 2022"""
import numpy as np
import torch

from rrc_2022_datasets import PolicyBase
from rrc_2022_datasets.utils import get_pose_from_keypoints

from . import policies

import pickle
from rl_games.algos_torch import players
from types import SimpleNamespace

import time

def scale_transform(x, lower, upper):
    """
    Normalizes a given input tensor to a range of [-1, 1].

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Normalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)

def unscale_transform(x, lower, upper):
    """
    Denormalizes a given input tensor from range of [-1, 1] to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Denormalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)[...,np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * q_w[...,np.newaxis] * 2.0
    c = q_vec * \
        np.matmul(q_vec.reshape((1, 3)), v.reshape((3, 1))).squeeze(-1) * 2.0
    return a - b + c

def quat_mul(a, b):
    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=-1)

    return quat

def quat_conjugate(a):
    return np.concatenate((-a[:3], a[-1:]), axis=-1)

class TorchBasePolicy(PolicyBase):
    # TODO: check if the order of fingers are same as simulator
    quats_symmetry = np.array([[0,0,0,1],[0, 0, 0.8660254, -0.5],[0, 0, 0.8660254, 0.5]], dtype=float)
    symm_agents_inds = [
        [0,1,2],
        [3,4,5],
        [6,7,8]
    ]
    _robot_limits: dict = {
        "joint_position": SimpleNamespace(
            # matches those on the real robot
            low=np.array([-0.33, 0.0, -2.7] * 3, dtype=np.float32),
            high=np.array([1.0, 1.57, 0.0] * 3, dtype=np.float32),
            default=np.array([0.0, 0.9, -2.0] * 3, dtype=np.float32),
        ),
        "joint_velocity": SimpleNamespace(
            low=np.full(9, -10, dtype=np.float32),
            high=np.full(9, 10, dtype=np.float32),
            default=np.zeros(9, dtype=np.float32),
        ),
        "joint_torque": SimpleNamespace(
            low=np.full(9, -0.36, dtype=np.float32),
            high=np.full(9, 0.36, dtype=np.float32),
            default=np.zeros(9, dtype=np.float32),
        ),
        # used if we want to have joint stiffness/damping as parameters`
        # "joint_stiffness": SimpleNamespace(
        #     low=np.array([1.0, 1.0, 1.0] * 3, dtype=np.float32),
        #     high=np.array([50.0, 50.0, 50.0] * 3, dtype=np.float32),
        # ),
        # "joint_damping": SimpleNamespace(
        #     low=np.array([0.01, 0.03, 0.0001] * 3, dtype=np.float32),
        #     high=np.array([1.0, 3.0, 0.01] * 3, dtype=np.float32),
        # ),
    }
    # limits of the object (mapped later: str -> torch.tensor)
    _object_limits: dict = {
        "position": SimpleNamespace(
            low=np.array([-0.3, -0.3, 0], dtype=np.float32),
            high=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            default=np.array([0, 0, 0.0325], dtype=np.float32)
        ),
        "orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
    }
    def __init__(
        self,
        torch_model_path,
        action_space,
        observation_space,
        episode_length,
    ):
        self.action_space = action_space
        self.device = "cpu"

        # get absolute path for predefined info
        checkpoint_path = policies.get_model_path("Trifinger-noMoveCost-masa-85.pt")
        params_path = policies.get_model_path("params_masa.pt")
        env_info_path = policies.get_model_path("env_info.pt")

        with open(params_path, "rb") as file:
            params = pickle.load(file)
        params['env_info_path'] = env_info_path
        self.agent = players.PpoPlayerContinuous(params)
        self.agent.restore(checkpoint_path)

        self.obs_limbs = np.zeros((3,9))
        self.obs_rotates = np.zeros((3,41))
        self.obs_center_rotate = np.zeros((3,14))

        # # load torch script
        # self.policy = torch.jit.load(
        #     torch_model_path, map_location=torch.device(self.device)
        # )

        # change constant buffers from numpy/lists into torch tensors
        # # limits for robot
        # for limit_name in self._robot_limits:
        #     # extract limit simple-namespace
        #     limit_dict = self._robot_limits[limit_name].__dict__
        #     # iterate over namespace attributes
        #     for prop, value in limit_dict.items():
        #         limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        # # limits for the object
        # for limit_name in self._object_limits:
        #     # extract limit simple-namespace
        #     limit_dict = self._object_limits[limit_name].__dict__
        #     # iterate over namespace attributes
        #     for prop, value in limit_dict.items():
        #         limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)

        self._dof_position_scale = SimpleNamespace(
                low=self._robot_limits["joint_position"].low,
                high=self._robot_limits["joint_position"].high
            )
        self._dof_velocity_scale = SimpleNamespace(
                low=self._robot_limits["joint_velocity"].low,
                high=self._robot_limits["joint_velocity"].high
            )
        
        object_obs_low = np.concatenate([
                                       self._object_limits["position"].low,
                                       self._object_limits["orientation"].low,
                                   ]*2)
        object_obs_high = np.concatenate([
                                        self._object_limits["position"].high,
                                        self._object_limits["orientation"].high,
                                    ]*2)
        self._object_obs_scale = SimpleNamespace(
                low=object_obs_low,
                high=object_obs_high
            )
        self._action_scale = SimpleNamespace(
            low=self._robot_limits["joint_torque"].low,
            high=self._robot_limits["joint_torque"].high
        )

    @staticmethod
    def is_using_flattened_observations():
        return False

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        # time1=time.time()
        obs = torch.from_numpy(self.get_obs(observation))
        obs = obs.float()
        # print(f"get obs time: {time.time()-time1}")
        action = self.agent.get_action(obs, is_determenistic = True).squeeze(0)
        action = unscale_transform(
                action,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        action = action.detach().numpy()
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        # print(f"forward time: {time.time()-time1}")
        return action
    
    def _get_obs_parts(self,observation):
        dof_pos_scaled = scale_transform(observation['robot_observation']['position'],
                                         self._dof_position_scale.low,
                                         self._dof_position_scale.high)
        dof_vel_scaled = scale_transform(observation['robot_observation']['velocity'],
                                         self._dof_velocity_scale.low,
                                         self._dof_velocity_scale.high)
        action_scaled = scale_transform(observation['action'],
                                        self._action_scale.low,
                                        self._action_scale.high)
        object_goal_position, object_goal_orientation = get_pose_from_keypoints(observation['desired_goal']['keypoints'])
        obs_center = np.concatenate(
            (
            observation['object_observation']['position'],
            observation['object_observation']['orientation'],
            object_goal_position,
            object_goal_orientation
            ),axis=-1
        )
        # obs_center_pos = torch.stack(
        #     (
        #     torch.from_numpy(observation['object_observation']['position']),
        #     torch.from_numpy(object_goal_position),
        #     ),dim=0
        # )
        # obs_center_ori = torch.stack(
        #     (
        #     torch.from_numpy(observation['object_observation']['orientation']),
        #     torch.from_numpy(object_goal_orientation)
        #     ),dim=0
        # )

        return dof_pos_scaled, dof_vel_scaled, action_scaled, obs_center
    
    def get_obs(self, observation):
        dof_pos_scaled, dof_vel_scaled, action_scaled, obs_center = self._get_obs_parts(observation)
        # stack observation for symmetric parts of the robot
        for j in range(3):
            self.obs_limbs[j, :3]=dof_pos_scaled[self.symm_agents_inds[j]]
            self.obs_limbs[j, 3:6]=dof_vel_scaled[self.symm_agents_inds[j]]
            self.obs_limbs[j, 6:]=action_scaled[self.symm_agents_inds[j]]

        for i in range(3):
            
            self.obs_center_rotate[i] = obs_center
            self.obs_center_rotate[i,:3] = quat_rotate_inverse(self.quats_symmetry[i], self.obs_center_rotate[i,:3])
            self.obs_center_rotate[i,3:7] = quat_mul(quat_conjugate(self.quats_symmetry[i]), self.obs_center_rotate[i,3:7])
            self.obs_center_rotate[i,7:10] = quat_rotate_inverse(self.quats_symmetry[i], self.obs_center_rotate[i,7:10])
            self.obs_center_rotate[i,10:] = quat_mul(quat_conjugate(self.quats_symmetry[i]), self.obs_center_rotate[i,10:])
            self.obs_center_rotate[i] = scale_transform(self.obs_center_rotate[i],
                                                self._object_obs_scale.low,
                                                self._object_obs_scale.high)
            self.obs_rotates[i,:14]=self.obs_center_rotate[i]
            self.obs_rotates[i,14:23]=self.obs_limbs[i]
            self.obs_rotates[i,23:32]=self.obs_limbs[(i+1)%3]
            self.obs_rotates[i,32:]=self.obs_limbs[(i+2)%3]
        return self.obs_rotates

    # def process_obs(self, obs):
    #     """ generate obs for masa given dict obs
    #     """



class TorchPushPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model = policies.get_model_path("push.pt")
        super().__init__(model, action_space, observation_space, episode_length)


class TorchLiftPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model = None #policies.get_model_path("lift.pt")
        super().__init__(model, action_space, observation_space, episode_length)
