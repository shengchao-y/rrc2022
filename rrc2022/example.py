"""Example policy for Real Robot Challenge 2022"""
import numpy as np
import torch

from rrc_2022_datasets import PolicyBase
from rrc_2022_datasets.utils import get_pose_from_keypoints

from . import policies

import pickle
from rl_games.algos_torch import players
from types import SimpleNamespace

def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
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

def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
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
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

class TorchBasePolicy(PolicyBase):
    # TODO: check if the order of fingers are same as simulator
    quats_symmetry = torch.tensor([[[0,0,0,1]],[[0, 0, 0.8660254, -0.5]],[[0, 0, 0.8660254, 0.5]]], dtype=float)
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

        checkpoint_path = "rrc2022/Trifinger-noMoveCost-masa-85.pth"
        with open("rrc2022/params_masa.pk", "rb") as file:
            params = pickle.load(file)
        self.agent = players.PpoPlayerContinuous(params)
        self.agent.restore(checkpoint_path)

        # # load torch script
        # self.policy = torch.jit.load(
        #     torch_model_path, map_location=torch.device(self.device)
        # )

        # change constant buffers from numpy/lists into torch tensors
        # limits for robot
        for limit_name in self._robot_limits:
            # extract limit simple-namespace
            limit_dict = self._robot_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        # limits for the object
        for limit_name in self._object_limits:
            # extract limit simple-namespace
            limit_dict = self._object_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)

        self._dof_position_scale = SimpleNamespace(
                low=self._robot_limits["joint_position"].low,
                high=self._robot_limits["joint_position"].high
            )
        self._dof_velocity_scale = SimpleNamespace(
                low=self._robot_limits["joint_velocity"].low,
                high=self._robot_limits["joint_velocity"].high
            )
        
        object_obs_low = torch.cat([
                                       self._object_limits["position"].low,
                                       self._object_limits["orientation"].low,
                                   ]*2)
        object_obs_high = torch.cat([
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
        obs = self.get_obs(observation).squeeze(0)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action = self.agent.get_action(obs, is_determenistic = True).squeeze(0)
        action = unscale_transform(
                action,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        action = action.detach().numpy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
    
    def _get_obs_parts(self,observation):
        dof_pos_scaled = scale_transform(torch.from_numpy(observation['robot_observation']['position']),
                                         self._dof_position_scale.low,
                                         self._dof_position_scale.high)
        dof_vel_scaled = scale_transform(torch.from_numpy(observation['robot_observation']['velocity']),
                                         self._dof_velocity_scale.low,
                                         self._dof_velocity_scale.high)
        action_scaled = scale_transform(torch.from_numpy(observation['action']),
                                        self._action_scale.low,
                                        self._action_scale.high)
        object_goal_position, object_goal_orientation = get_pose_from_keypoints(observation['desired_goal']['keypoints'])
        obs_center = torch.cat(
            (
            torch.from_numpy(observation['object_observation']['position']),
            torch.from_numpy(observation['object_observation']['orientation']),
            torch.from_numpy(object_goal_position),
            torch.from_numpy(object_goal_orientation)
            ),dim=-1
        )

        return dof_pos_scaled.unsqueeze(0), dof_vel_scaled.unsqueeze(0), action_scaled.unsqueeze(0), obs_center.unsqueeze(0)
    
    def get_obs(self, observation):
        dof_pos_scaled, dof_vel_scaled, action_scaled, obs_center = self._get_obs_parts(observation)
        # stack observation for symmetric parts of the robot
        obs_limbs = []
        for i in range(3):
            obs_limbs.append(torch.cat(
                (dof_pos_scaled[:,self.symm_agents_inds[i]],
                 dof_vel_scaled[:,self.symm_agents_inds[i]],
                 action_scaled[:,self.symm_agents_inds[i]],), dim=-1
            ))
        obs_rotates = []
        for i in range(3):
            obs_center_rotate = obs_center.clone()
            obs_center_rotate[:,:3] = quat_rotate_inverse(self.quats_symmetry[i], obs_center_rotate[:,:3])
            obs_center_rotate[:,3:7] = quat_mul(quat_conjugate(self.quats_symmetry[i]), obs_center_rotate[:,3:7])
            obs_center_rotate[:,7:10] = quat_rotate_inverse(self.quats_symmetry[i], obs_center_rotate[:,7:10])
            obs_center_rotate[:,10:] = quat_mul(quat_conjugate(self.quats_symmetry[i]), obs_center_rotate[:,10:])
            obs_center_rotate = scale_transform(obs_center_rotate,
                                                self._object_obs_scale.low,
                                                self._object_obs_scale.high)
            obs_rotates.append(torch.cat(
                (obs_center_rotate,
                 obs_limbs[i],
                 obs_limbs[(i+1)%3],
                 obs_limbs[(i+2)%3]), dim=-1
            ))
        return torch.stack(obs_rotates, dim=1)

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
        model = policies.get_model_path("lift.pt")
        super().__init__(model, action_space, observation_space, episode_length)
