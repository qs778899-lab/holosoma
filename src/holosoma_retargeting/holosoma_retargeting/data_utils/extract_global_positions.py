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
        
        #! foot: Define offsets for Nokov End Sites (Toe) relative to Foot (Ankle)
        # Assuming offset is constant for Nokov skeleton: (0, -10, 15.12)
        # Note: These values come from snooker2.bvh
        left_toe_offset = np.array([0.0, -10.0, 15.12]) 
        right_toe_offset = np.array([0.0, -10.0, 15.12])
        
        for joint in target_joints:
            if joint in joint_names:
                idx = joint_names.index(joint)
                extracted_positions.append(positions[:, idx])
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
            
            # 处理左手架杆点：直接映射真人的左手全局位置
            elif joint == "LeftHandBridge" and "LeftHand" in joint_names:
                idx = joint_names.index("LeftHand")
                extracted_positions.append(positions[:, idx])
            # 处理右手握杆点：直接映射真人的右手全局位置
            elif joint == "RightHandGrip" and "RightHand" in joint_names:
                idx = joint_names.index("RightHand")
                extracted_positions.append(positions[:, idx])
            # 处理虚拟球杆尖端：通过真人左右手的位置“延长”出球杆末端
            elif joint == "CueTip" and "RightHand" in joint_names and "LeftHand" in joint_names:
                # Calculate Cue Tip position: extend from RightHand through LeftHand
                # 获取真人右手和左手在 BVH 中的索引
                rh_idx = joint_names.index("RightHand")
                lh_idx = joint_names.index("LeftHand")
                # 提取所有帧的左右手世界坐标
                rh_pos = positions[:, rh_idx]
                lh_pos = positions[:, lh_idx]    
                # 计算从右手指向左手的向量（即球杆的方向）
                direction = lh_pos - rh_pos
                dist = np.linalg.norm(direction, axis=-1, keepdims=True)
                # Normalize direction
                direction_norm = direction / (dist + 1e-8)
                
                # 优化方案：不再使用1.35m的总长。
                # 我们定义尖端在左手(架杆点)前方 0.4 米处。
                # 这样缩短了力臂，能显著减少重定向时机器人手腕的异常扭动。
                cue_tip_pos = lh_pos + direction_norm * 0.4
                extracted_positions.append(cue_tip_pos)
            else:
                print(f"Warning: Joint {joint} not found in BVH. Using zeros.")
                extracted_positions.append(np.zeros((positions.shape[0], 3)))

        
        # extracted_positions： 存储了经过筛选、排序以及计算后的关节点全局坐标数据。
        positions = np.stack(extracted_positions, axis=1)
        joint_names = target_joints

    return {
        "positions": positions,
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
        np.save(str(output_npy), result["positions"])
        print(f"  Saved to: {output_npy}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
