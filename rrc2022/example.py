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
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)

def unscale_transform(x, lower, upper):
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)[...,np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * q_w[...,np.newaxis] * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    # np.matmul(q_vec.reshape((1, 3)), v.reshape((3, 1))).squeeze(-1) * 2.0
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

    # quat = np.stack([x, y, z, w], axis=-1)

    return np.array([x,y,z,w])

def quat_conjugate(a):
    return np.concatenate((-a[:3], a[-1:]), axis=-1)

class TorchBasePolicy(PolicyBase):
    # TODO: check if the order of fingers are same as simulator
    quats_symmetry = np.array([[0,0,0,1],[0, 0, 0.8660254, -0.5],[0, 0, 0.8660254, 0.5]], dtype=float)
    quats_symmetry_conjugate = quats_symmetry.copy()
    for i in range(3):
        quats_symmetry_conjugate[i] = quat_conjugate(quats_symmetry_conjugate[i])
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
        obs_test_path = policies.get_model_path("obs_test.pt")
        with open(obs_test_path, 'rb') as file:
            obs_test = pickle.load(file)

        with open(params_path, "rb") as file:
            params = pickle.load(file)
        params['env_info_path'] = env_info_path
        self.agent = players.PpoPlayerContinuous(params)
        self.agent.restore(checkpoint_path)

        self.obs_limbs = np.zeros((3,9))
        self.obs_rotates = np.zeros((3,41))
        self.obs_center_rotate = np.zeros((3,14))

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

        self.keypoints = None
        self.n_step = 0
        self.get_action(obs_test)

    @staticmethod
    def is_using_flattened_observations():
        return False

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        # time1=time.time()
        with torch.no_grad():
            obs = torch.from_numpy(self.get_obs(observation))
            obs = obs.float()
            # print(f"get obs time: {time.time()-time1}")
            # time2=time.time()
            action = self.agent.get_action(obs, is_determenistic = True).squeeze(0)
            action = unscale_transform(
                    action,
                    lower=self._action_scale.low,
                    upper=self._action_scale.high
                )
            action = action.detach().numpy()
            # action = np.clip(action, self.action_space.low, self.action_space.high)
            # print(f"forward time: {time.time()-time2}")
            return action
    
    def get_obs(self, observation):
        dof_pos_scaled = scale_transform(observation['robot_observation']['position'],
                                         self._dof_position_scale.low,
                                         self._dof_position_scale.high)
        dof_vel_scaled = scale_transform(observation['robot_observation']['velocity'],
                                         self._dof_velocity_scale.low,
                                         self._dof_velocity_scale.high)
        action_scaled = scale_transform(observation['action'],
                                        self._action_scale.low,
                                        self._action_scale.high)

        # 1500 steps for one episode
        if self.n_step%1500 != 0:
            pass
        else:
            self.object_goal_position, self.object_goal_orientation = get_pose_from_keypoints(observation['desired_goal']['keypoints'])
            self.keypoints = observation['desired_goal']['keypoints']
            self.goal_pos_rotate = [self.object_goal_position, quat_rotate_inverse(self.quats_symmetry[1], self.object_goal_position), quat_rotate_inverse(self.quats_symmetry[2], self.object_goal_position)]
            self.goal_ori_rotate = [self.object_goal_orientation, quat_mul(self.quats_symmetry_conjugate[1], self.object_goal_orientation), quat_mul(self.quats_symmetry_conjugate[2], self.object_goal_orientation)]
        self.n_step+=1

        for j in range(3):
            self.obs_limbs[j, :3]=dof_pos_scaled[self.symm_agents_inds[j]]
            self.obs_limbs[j, 3:6]=dof_vel_scaled[self.symm_agents_inds[j]]
            self.obs_limbs[j, 6:]=action_scaled[self.symm_agents_inds[j]]

        self.obs_center_rotate[0,:3] = observation['object_observation']['position']
        self.obs_center_rotate[0,3:7] = observation['object_observation']['orientation']
        self.obs_center_rotate[0,7:10] = self.goal_pos_rotate[0]
        self.obs_center_rotate[0,10:] = self.goal_ori_rotate[0]
        self.obs_center_rotate[0] = scale_transform(self.obs_center_rotate[0],
                                            self._object_obs_scale.low,
                                            self._object_obs_scale.high)
        self.obs_rotates[0,:14]=self.obs_center_rotate[0]
        self.obs_rotates[0,14:23]=self.obs_limbs[0]
        self.obs_rotates[0,23:32]=self.obs_limbs[1]
        self.obs_rotates[0,32:]=self.obs_limbs[2]

        for i in range(1,3):
            self.obs_center_rotate[i,:3] = quat_rotate_inverse(self.quats_symmetry[i], observation['object_observation']['position'])
            self.obs_center_rotate[i,3:7] = quat_mul(self.quats_symmetry_conjugate[i], observation['object_observation']['orientation'])
            self.obs_center_rotate[i,7:10] = self.goal_pos_rotate[i]
            self.obs_center_rotate[i,10:] = self.goal_ori_rotate[i]
            self.obs_center_rotate[i] = scale_transform(self.obs_center_rotate[i],
                                                self._object_obs_scale.low,
                                                self._object_obs_scale.high)
            self.obs_rotates[i,:14]=self.obs_center_rotate[i]
            self.obs_rotates[i,14:23]=self.obs_limbs[i]
            self.obs_rotates[i,23:32]=self.obs_limbs[(i+1)%3]
            self.obs_rotates[i,32:]=self.obs_limbs[(i+2)%3]
        return self.obs_rotates

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