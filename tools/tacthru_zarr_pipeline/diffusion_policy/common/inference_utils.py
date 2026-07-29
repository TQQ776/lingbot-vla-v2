import collections
from typing import Dict, List, Tuple

import numpy as np
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_util import mat_to_pose, mat_to_pose10d, pose10d_to_mat, pose_to_mat, rotvec_to_rot6d
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep


def get_real_obs_resolution(shape_meta: dict) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            co, ho, wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(env_obs: Dict[str, np.ndarray], shape_meta: dict) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = env_obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)
        elif type == "low_dim":
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_umi_obs_dict(
    env_obs: Dict[str, np.ndarray],
    shape_meta: dict,
    obs_pose_repr: str = "relative",
    tx_robot1_robot0: np.ndarray = None,
    episode_start_pose: List[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    n_robots = 1
    obs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta["obs"]
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = env_obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)
        elif type in ["tac_rgb", "tac_depth", "tac_shear"] and not key.endswith("marker"):
            this_imgs_in = env_obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)
        elif (type == "low_dim" and ("eef" not in key)) or key.endswith("marker"):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # handle multi-robots
            ks = key.split("_")
            if ks[0].startswith("robot"):
                robot_prefix_map[ks[0]].append(key)

    # generate relative pose
    for robot_prefix in robot_prefix_map.keys():
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([env_obs[robot_prefix + "_eef_pos"], env_obs[robot_prefix + "_eef_rot_axis_angle"]], axis=-1))

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(pose_mat, base_pose_mat=pose_mat[-1], pose_rep=obs_pose_repr, backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + "_eef_pos"] = obs_pose[..., :3]
        obs_dict_np[robot_prefix + "_eef_rot_axis_angle"] = obs_pose[..., 3:]

    # generate relative pose with respect to episode start
    if episode_start_pose is not None:
        for robot_id in range(n_robots):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([env_obs[f"robot{robot_id}_eef_pos"], env_obs[f"robot{robot_id}_eef_rot_axis_angle"]], axis=-1))

            # get start pose
            start_pose = episode_start_pose[robot_id]
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(pose_mat, base_pose_mat=start_pose_mat, pose_rep="relative", backward=False)

            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict_np[f"robot{robot_id}_eef_rot_axis_angle_wrt_start"] = rel_obs_pose[:, 3:]
    else:
        for robot_id in range(n_robots):
            obs_dict_np[f"robot{robot_id}_eef_rot_axis_angle_wrt_start"] = rotvec_to_rot6d(env_obs[f"robot{robot_id}_eef_rot_axis_angle_wrt_start"])

    return obs_dict_np


def get_real_umi_action(action: np.ndarray, env_obs: Dict[str, np.ndarray], action_pose_repr: str = "relative"):
    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(
            np.concatenate([env_obs[f"robot{robot_idx}_eef_pos"][-1], env_obs[f"robot{robot_idx}_eef_rot_axis_angle"][-1]], axis=-1)
        )

        start = robot_idx * 10
        action_pose10d = action[..., start : start + 9]
        action_grip = action[..., start + 9 : start + 10]
        action_pose_mat = pose10d_to_mat(action_pose10d)  # [x, y, z, rot6d] → 4x4 pose matrix]

        # solve relative action
        action_mat = convert_pose_mat_rep(action_pose_mat, base_pose_mat=pose_mat, pose_rep=action_pose_repr, backward=True)

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action
