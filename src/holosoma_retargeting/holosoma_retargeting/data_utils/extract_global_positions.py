#!/usr/bin/env python3
"""
Simple script to extract global positions from LAFAN dataset BVH files.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from lafan1 import extract, utils  # type: ignore[import-not-found]
from scipy.spatial.transform import Rotation as R

# Use absolute imports if possible or relative if running as module
import sys
import os
from pathlib import Path

# Add the package root to path to import config_types
# This script is in src/holosoma_retargeting/holosoma_retargeting/data_utils/
# Package root is src/holosoma_retargeting/
package_root = Path(__file__).resolve().parents[2]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

try:
    from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY
except ImportError:
    # Fallback if not installed or path not set correctly
    DEMO_JOINTS_REGISTRY = {}


def extract_global_positions(bvh_file_path, target_joints=None):
    """
    Extract global positions from a BVH file.

    Args:
        bvh_file_path (str): Path to the BVH file
        target_joints (list, optional): List of joint names to extract in specific order.
                                       If None, extracts all joints from BVH.

    Returns:
        dict: Dictionary containing:
            - 'positions': numpy array of shape (frames, joints, 3) with global positions
            - 'joint_names': list of joint names
            - 'parents': list of parent indices
            - 'num_frames': number of frames
            - 'num_joints': number of joints
    """
    # Read BVH file
    anim = extract.read_bvh(bvh_file_path)

    # Compute global positions using Forward Kinematics：利用原始数据的局部位置和旋转计算每个关节的全局位置和旋转
    global_quats, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    
    positions = global_positions / 100 # cm to m
    joint_names = list(anim.bones)
    
    if target_joints:
        # Extract only target joints in the specified order
        extracted_positions = []
        extracted_quats = [] #! cube: 新增旋转提取
        
        #! foot: Define offsets for Nokov End Sites (Toe) relative to Foot (Ankle)
        # Assuming offset is constant for Nokov skeleton: (0, -10, 15.12)
        # Note: These values come from snooker2.bvh
        left_toe_offset = np.array([0.0, -10.0, 15.12]) 
        right_toe_offset = np.array([0.0, -10.0, 15.12])
        
        for joint in target_joints:
            if joint in joint_names:
                idx = joint_names.index(joint)
                extracted_positions.append(positions[:, idx])
                extracted_quats.append(global_quats[:, idx]) #! cube: 提取旋转 (w,x,y,z)
            elif joint == "LeftFootMod" and "LeftFoot" in joint_names:
                # Special case for Nokov foot mod - compute Toe position
                # BVH文件中没有LeftFootMod，这是我们为了约束脚尖而虚拟出来的关节
                # 计算逻辑：LeftFoot全局位置 + LeftFoot全局旋转 * 偏移量
                idx = joint_names.index("LeftFoot")

                #! foot :
                # extracted_positions.append(positions[:, idx])
                foot_pos = positions[:, idx] # (Frames, 3)
                foot_quat = global_quats[:, idx] # (Frames, 4) (w, x, y, z)
                # Convert (w, x, y, z) to (x, y, z, w) for scipy Rotation
                foot_quat_scipy = np.roll(foot_quat, -1, axis=1)
                # Rotate offset
                r = R.from_quat(foot_quat_scipy)
                rotated_offset = r.apply(left_toe_offset)
                # Add offset (scaled by 1/100 as positions are divided by 100)
                toe_pos = foot_pos + (rotated_offset / 100.0)
                extracted_positions.append(toe_pos)
                extracted_quats.append(foot_quat) #! cube: 脚尖旋转暂用脚踝旋转
                
                
            elif joint == "RightFootMod" and "RightFoot" in joint_names:
                # Special case for Nokov foot mod - compute Toe position
                # BVH文件中没有RightFootMod，这是我们为了约束脚尖而虚拟出来的关节
                # 计算逻辑：RightFoot全局位置 + RightFoot全局旋转 * 偏移量
                idx = joint_names.index("RightFoot")

                #! foot :
                # extracted_positions.append(positions[:, idx])
                foot_pos = positions[:, idx]
                foot_quat = global_quats[:, idx]
                # Convert (w, x, y, z) to (x, y, z, w) for scipy Rotation
                foot_quat_scipy = np.roll(foot_quat, -1, axis=1)
                # Rotate offset
                r = R.from_quat(foot_quat_scipy)
                rotated_offset = r.apply(right_toe_offset)
                # Add offset (scaled by 1/100)
                toe_pos = foot_pos + (rotated_offset / 100.0)
                extracted_positions.append(toe_pos)
                extracted_quats.append(foot_quat) #! cube: 脚尖旋转暂用脚踝旋转
            
            # # 处理左手架杆点：考虑手部相对于手腕的偏移
            # elif joint == "LeftHandBridge" and "LeftHand" in joint_names:
            #     idx = joint_names.index("LeftHand")
            #     lh_pos = positions[:, idx]
            #     lh_quat = global_quats[:, idx]
                
            #     # 1. 获取 parent (LeftHand) 的全局旋转矩阵
            #     r_parent = R.from_quat(np.roll(lh_quat, -1, axis=1))
                
            #     # 2. XML: left_hand_bridge site pos="0.0415 0.003 0" quat="1 0 0 0" (Identity)
            #     # 定义局部变换：平移 offset 和 旋转 r_local
            #     offset = np.array([0.0415, 0.003, 0])
            #     r_local = R.identity() #单位旋转矩阵
                
            #     # 3. 精确计算全局位置和旋转 (R_global = R_parent * R_local)
            #     extracted_positions.append(lh_pos + r_parent.apply(offset))
            #     r_global = r_parent * r_local
            #     # 转回 wxyz 格式保存
            #     q_wxyz = np.roll(r_global.as_quat(), 1, axis=-1)
            #     extracted_quats.append(q_wxyz) #? 为什么是wxyz？

            # # 处理右手握杆点：考虑手部相对于手腕的偏移
            # elif joint == "RightHandGrip" and "RightHand" in joint_names:
            #     idx = joint_names.index("RightHand")
            #     rh_pos = positions[:, idx]
            #     rh_quat = global_quats[:, idx]
                
            #     # 1. 获取 parent (RightHand) 的全局旋转矩阵
            #     r_parent = R.from_quat(np.roll(rh_quat, -1, axis=1))
                
            #     # 2. XML: right_hand_grip site pos="0.0415 -0.003 0" quat="1 0 0 0" (Identity)
            #     offset = np.array([0.0415, -0.003, 0])
            #     r_local = R.identity() #单位旋转矩阵
                
            #     # 3. 精确计算全局位置和旋转
            #     extracted_positions.append(rh_pos + r_parent.apply(offset))
            #     r_global = r_parent * r_local
            #     q_wxyz = np.roll(r_global.as_quat(), 1, axis=-1)
            #     extracted_quats.append(q_wxyz)

            # # 处理虚拟球杆尖端：从“球杆上的握持点”开始延伸
            # elif joint == "CueTip" and "RightHand" in joint_names and "LeftHand" in joint_names:
            #     rh_idx = joint_names.index("RightHand")
            #     lh_idx = joint_names.index("LeftHand")
                
            #     # 1. 基础位姿提取
            #     rh_pos = positions[:, rh_idx]
            #     rh_quat = global_quats[:, rh_idx]
            #     r_rh = R.from_quat(np.roll(rh_quat, -1, axis=1))
                
            #     # 2. 找到准确的握持参考点和桥手点 (基于 XML 偏移)
            #     cue_grip_on_stick = rh_pos + r_rh.apply(np.array([0.1215, 0.017, 0.0]))
                
            #     lh_pos = positions[:, lh_idx]
            #     lh_quat = global_quats[:, lh_idx]
            #     r_lh = R.from_quat(np.roll(lh_quat, -1, axis=1))
            #     bridge_pos = lh_pos + r_lh.apply(np.array([0.0415, 0.003, 0.0]))
                
            #     # 3. 计算方向向量并延伸 (从 cue_grip_on_stick 到桥手点)
            #     direction = bridge_pos - cue_grip_on_stick
            #     direction_norm = direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
                
            #     # 4. 根据 XML 计算固定杆尖位置
            #     fixed_cue_length = 1.075
            #     cue_tip_pos = cue_grip_on_stick + direction_norm * fixed_cue_length
            #     extracted_positions.append(cue_tip_pos)
                
            #     # 5. 精确重构球杆的全局旋转矩阵 (Alignment Rotation)
            #     # 原理：将局部 Z 轴对齐到瞄准方向，同时通过右手腕的 X 轴确定扭转(Roll)
            #     z_axes = direction_norm
            #     # 提取右手腕当前的全局 X 轴作为参考
            #     rh_mats = r_rh.as_matrix()
            #     x_refs = rh_mats[:, :, 0] 
                
            #     # Gram-Schmidt 正交化
            #     y_axes = np.cross(z_axes, x_refs)
            #     y_axes /= (np.linalg.norm(y_axes, axis=-1, keepdims=True) + 1e-8)
            #     x_axes = np.cross(y_axes, z_axes)
                
            #     # 构造旋转矩阵 [x, y, z] 并转为四元数
            #     cue_mats = np.stack([x_axes, y_axes, z_axes], axis=-1)
            #     r_cue = R.from_matrix(cue_mats)
            #     q_wxyz = np.roll(r_cue.as_quat(), 1, axis=-1)
            #     extracted_quats.append(q_wxyz) #? 还未确认计算过程是否正确，但目前用不上
            # else:
            #     print(f"Warning: Joint {joint} not found in BVH. Using zeros.")
            #     extracted_positions.append(np.zeros((positions.shape[0], 3)))
            #     extracted_quats.append(np.array([1.0, 0.0, 0.0, 0.0] * positions.shape[0]).reshape(-1, 4))

        
        # extracted_positions： 存储了经过筛选、排序以及计算后的关节点全局坐标数据。
        positions = np.stack(extracted_positions, axis=1)
        quats = np.stack(extracted_quats, axis=1) # (Frames, Joints, 4)
        # 合并为 7D 数据 (pos, quat)
        combined_data = np.concatenate([positions, quats], axis=-1) # (Frames, Joints, 7)
        joint_names = target_joints
    else:
        # 如果没有 target_joints，默认只保存 positions 以保持兼容
        combined_data = positions

    return {
        "combined_data": combined_data,
        "joint_names": joint_names,
        "parents": anim.parents, # Note: parents indices will be wrong if joints are reordered
        "num_frames": positions.shape[0],
        "num_joints": positions.shape[1],
    }



def save_global_positions_to_npy(global_positions, output_path):
    """
    Save global positions to a .npy file.

    Args:
        global_positions (numpy.ndarray): Global positions array
        output_path (str): Output file path
    """
    np.save(output_path, global_positions)
    print(f"Saved global positions to: {output_path}")


@dataclass
class Config:
    """Configuration for extracting global positions from BVH files."""

    input_dir: str = "./lafan1/lafan"
    output_dir: str = "../demo_data/lafan"
    data_format: str = "lafan"  #? 这里的data_format参数的作用是？ Data format from DEMO_JOINTS_REGISTRY


def main(cfg: Config):
    """
    Main function to extract global positions from BVH files.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {cfg.input_dir} not found!")
        return

    # Get target joints if data_format is specified
    target_joints = DEMO_JOINTS_REGISTRY.get(cfg.data_format)
    if target_joints:
        print(f"Using target joints from format: {cfg.data_format}")
    else:
        print(f"Warning: Format {cfg.data_format} not found in registry. Extracting all joints.")

    # Get list of BVH files
    bvh_files = [f.name for f in input_dir.iterdir() if f.is_file() and f.suffix == ".bvh"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each BVH file
    for bvh_file in bvh_files:
        print(f"\nProcessing: {bvh_file}")

        bvh_path = input_dir / bvh_file

        # Extract global positions
        result = extract_global_positions(str(bvh_path), target_joints=target_joints)

        print(f"  Frames: {result['num_frames']}")
        print(f"  Joints: {result['num_joints']}")

        # Save to .npy file
        output_npy = output_dir / f"{bvh_file[:-4]}.npy"
        np.save(str(output_npy), result["combined_data"])
        print(f"  Saved to: {output_npy}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
