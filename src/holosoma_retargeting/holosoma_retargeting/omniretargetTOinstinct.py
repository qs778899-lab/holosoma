from __future__ import annotations

import numpy as np
import argparse
import mujoco
import os
import contextlib
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

'''
python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/omniretargetTOinstinct.py \
    --input /home/huangyucheng/桌面/Omniretarget/data/stairs_01_original.npz \
    --output /home/huangyucheng/桌面/Omniretarget/data_instinct/stairs_01_original_retargeted.npz \
    --source-xml /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml \
    --target-urdf /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_torsobase_popsicle.urdf \
    --terrain-scale 1

python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/omniretargetTOinstinct.py \
    --input /home/huangyucheng/桌面/Omniretarget/data \
    --output /home/huangyucheng/桌面/Omniretarget/data_instinct \
    --source-xml /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml \
    --target-urdf /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_torsobase_popsicle.urdf \
    --terrain-scale 1
    
说明：
  holosoma 原版 npz 格式 (qpos):
      shape = (T, 7 + DOF)
      列顺序 = [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, joint_0, ..., joint_N]
      (标准 MuJoCo qpos 格式)

  instinct 训练所需 retargeted npz 格式（由 amass_motion.py _read_retargetted_motion_file 读取）:
      framerate   scalar        — 帧率（注意：是 framerate，不是 fps）
      joint_names (DOF,)        — 关节名称（不含 freejoint，与 MuJoCo XML 关节顺序一致）
      joint_pos   (T, DOF)      — 纯关节角度（不含 base 的 7 列）
      base_pos_w  (T, 3)        — root 位置（世界坐标）
      base_quat_w (T, 4)        — root 四元数 wxyz（世界坐标）

  注意：转换时同时使用两个模型
      1) source_xml: 解释 holosoma 原始 qpos 的语义
      2) target_urdf: 与 instinct 训练/可视化实际使用的机器人一致
     这样可以按真实 target 模型反解 root 位姿，避免整体 z 高度偏差。

  地形缩放校正 (--terrain-scale):
      holosoma 在 retarget 时对地形 mesh 进行了统一缩放（见 box_assets.xml 中的 scale）。
      该缩放导致台阶高度比 instinct 中使用的 STL 地形矮。
      例如 scale=0.7416 时，台阶第 4 级在 holosoma 中为 0.460m，在 instinct 中为 0.620m，
      差距达 0.16m，造成可视化时机器人脚部陷入台阶。
      校正公式（逐帧）:
          terrain_z ≈ min(left_foot_z, right_foot_z)   # 用支撑脚 Z 近似地形高度
          new_z = terrain_z * (1/terrain_scale - 1) + z
      即：只把 "地形高度贡献" 从缩放空间还原到原始空间，机器人相对地形的高度不变。
      使用逐帧足底位置比固定 h_standing 更准确，能正确处理蹲下、过渡等非站立姿态。

  instinct 项目会自己做前向运动学和速度估计，不需要我们提供 body_pos_w 等字段。
  文件名必须以 "retargetted.npz" 或 "retargeted.npz" 结尾，
  instinct 才会走 _read_retargetted_motion_file 这个读取路径。
'''


@contextlib.contextmanager
def cd(newdir):
    """临时切换工作目录，用于 MuJoCo 加载带相对路径 mesh 的 XML 文件。"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def compose_transform(
    pos_ab: np.ndarray,
    quat_ab_wxyz: np.ndarray,
    pos_bc: np.ndarray,
    quat_bc_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose transforms T_ac = T_ab * T_bc."""
    rot_ab = Rotation.from_quat(quat_wxyz_to_xyzw(quat_ab_wxyz))
    rot_bc = Rotation.from_quat(quat_wxyz_to_xyzw(quat_bc_wxyz))
    pos_ac = pos_ab + rot_ab.apply(pos_bc)
    quat_ac = quat_xyzw_to_wxyz((rot_ab * rot_bc).as_quat())
    return pos_ac, quat_ac


def invert_transform(pos_ab: np.ndarray, quat_ab_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert transform T_ab -> T_ba."""
    rot_ab = Rotation.from_quat(quat_wxyz_to_xyzw(quat_ab_wxyz))
    rot_ba = rot_ab.inv()
    pos_ba = -rot_ba.apply(pos_ab)
    quat_ba = quat_xyzw_to_wxyz(rot_ba.as_quat())
    return pos_ba, quat_ba


def parse_target_urdf(target_urdf_path: str) -> tuple[list[str], list[dict], str]:
    """
    Parse target URDF and return:
    1) actuated joint names in target order
    2) root->pelvis joint chain
    3) root link name
    """
    xml_root = ET.parse(target_urdf_path).getroot()

    all_links = set()
    child_links = set()
    joints = []
    actuated_joint_names = []
    joint_by_child = {}

    for link in xml_root.findall("link"):
        name = link.get("name")
        if name:
            all_links.add(name)

    for joint in xml_root.findall("joint"):
        name = joint.get("name")
        joint_type = joint.get("type", "fixed")
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        origin_elem = joint.find("origin")
        axis_elem = joint.find("axis")

        parent_link = parent_elem.get("link") if parent_elem is not None else None
        child_link = child_elem.get("link") if child_elem is not None else None
        if child_link:
            child_links.add(child_link)

        xyz = np.fromstring(origin_elem.get("xyz", "0 0 0"), sep=" ", dtype=np.float64) if origin_elem is not None else np.zeros(3, dtype=np.float64)
        rpy = np.fromstring(origin_elem.get("rpy", "0 0 0"), sep=" ", dtype=np.float64) if origin_elem is not None else np.zeros(3, dtype=np.float64)
        axis = np.fromstring(axis_elem.get("xyz", "0 0 1"), sep=" ", dtype=np.float64) if axis_elem is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64)

        joint_info = {
            "name": name,
            "type": joint_type,
            "parent": parent_link,
            "child": child_link,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
        }
        joints.append(joint_info)
        if child_link:
            joint_by_child[child_link] = joint_info
        if joint_type not in ("fixed", "floating"):
            actuated_joint_names.append(name)

    root_candidates = sorted(all_links - child_links)
    if len(root_candidates) != 1:
        raise ValueError(f"Could not determine a unique root link from target URDF: {root_candidates}")
    root_link = root_candidates[0]

    # Build chain from root -> pelvis
    chain_rev = []
    current_link = "pelvis"
    while current_link != root_link:
        if current_link not in joint_by_child:
            raise ValueError(f"Cannot trace chain from pelvis to root '{root_link}' in target URDF.")
        joint_info = joint_by_child[current_link]
        chain_rev.append(joint_info)
        current_link = joint_info["parent"]
        if current_link is None:
            raise ValueError("Broken parent chain in target URDF while tracing pelvis->root.")

    root_to_pelvis_chain = list(reversed(chain_rev))
    return actuated_joint_names, root_to_pelvis_chain, root_link


def compute_root_to_pelvis_transform(
    root_to_pelvis_chain: list[dict],
    joint_positions_by_name: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute target transform from root link to pelvis using the target URDF chain."""
    pos = np.zeros(3, dtype=np.float64)
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for joint in root_to_pelvis_chain:
        origin_pos = joint["xyz"]
        origin_quat = quat_xyzw_to_wxyz(Rotation.from_euler("xyz", joint["rpy"]).as_quat())

        if joint["type"] in ("revolute", "continuous"):
            angle = joint_positions_by_name.get(joint["name"], 0.0)
            axis = joint["axis"].astype(np.float64)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis = axis / axis_norm
            joint_quat = quat_xyzw_to_wxyz(Rotation.from_rotvec(axis * angle).as_quat())
        else:
            joint_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        pos, quat_wxyz = compose_transform(pos, quat_wxyz, origin_pos, origin_quat)
        pos, quat_wxyz = compose_transform(
            pos,
            quat_wxyz,
            np.zeros(3, dtype=np.float64),
            joint_quat,
        )

    return pos, quat_wxyz


def convert_format(input_path: str, output_path: str, source_xml_path: str, target_urdf_path: str,
                    terrain_scale: float = 1.0):
    """
    将 holosoma 原版 omniretarget npz 转换为 instinct 训练所需的 retargeted 格式。

    holosoma 原版格式:
        qpos: (T, 7+DOF)  标准 MuJoCo qpos [pos(3), quat_wxyz(4), joints(DOF)]
        fps:  scalar

    instinct retargeted 格式（_read_retargetted_motion_file 读取）:
        framerate:   scalar
        joint_names: (DOF,)    目标 URDF 的 actuated joint 顺序
        joint_pos:   (T, DOF)  纯关节角，不含 base 的 7 列
        base_pos_w:  (T, 3)    root 位置
        base_quat_w: (T, 4)    root 四元数 wxyz

    terrain_scale: holosoma retarget 时对地形 mesh 施加的统一缩放因子。
        若 != 1.0，会对 base_pos_w 的 z 分量进行修正，补偿地形缩放差异。
    """
    print(f"[1/3] 加载原始 holosoma npz: {input_path}")
    data = np.load(input_path, allow_pickle=True)

    if 'qpos' not in data:
        print("错误: 输入文件中没有 'qpos' 字段，请确认是 holosoma/omniretarget 的输出文件。")
        return

    qpos = data['qpos'].astype(np.float32)   # (T, 7+DOF)
    T, nq = qpos.shape
    print(f"  qpos shape: {qpos.shape}")

    fps_raw = data.get('fps', 30.0)
    fps = float(fps_raw) if not isinstance(fps_raw, np.ndarray) else float(fps_raw.flat[0])
    print(f"  fps: {fps}")

    # ------------------------------------------------------------------
    # 2. 加载 MuJoCo XML 模型，提取关节名称（顺序与 qpos 列严格对应）
    # ------------------------------------------------------------------
    print(f"[2/3] 加载源 MuJoCo XML 模型: {source_xml_path}")
    xml_dir = str(Path(source_xml_path).parent)
    xml_name = Path(source_xml_path).name

    try:
        with cd(xml_dir):
            source_robot = mujoco.MjModel.from_xml_path(xml_name)
    except Exception as e:
        print(f"错误: 加载源 XML 失败: {e}")
        return

    print(f"  源模型 nq={source_robot.nq}, njnt={source_robot.njnt}")

    if source_robot.nq != nq:
        print(f"警告: 源 XML 模型 nq={source_robot.nq} 与 qpos 列数 {nq} 不一致！请确认使用了正确的模型文件。")

    # 源 joint 顺序与 qpos[7:] 严格对应
    source_joint_names = []
    for i in range(source_robot.njnt):
        if source_robot.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            source_joint_names.append(source_robot.joint(i).name)

    source_dof = len(source_joint_names)
    print(f"  源模型关节数 (不含 freejoint): {source_dof}")

    if source_dof != nq - 7:
        print(f"警告: 源模型关节数 {source_dof} 与 qpos 关节列数 {nq - 7} 不一致！")

    # 解析目标 URDF：获取 target joint 顺序和 root->pelvis 链
    target_joint_names, root_to_pelvis_chain, target_root_link = parse_target_urdf(target_urdf_path)
    print(f"  目标模型 root link: {target_root_link}")
    print(f"  目标模型关节数 (actuated): {len(target_joint_names)}")

    # ------------------------------------------------------------------
    # 3. 拆分 qpos，并按真实 target URDF 反解 root 位姿
    # ------------------------------------------------------------------
    print(f"[3/3] 拆分数据并保存: {output_path}")

    # 源 qpos: [pelvis_pos(3), pelvis_quat_wxyz(4), source_joints...]
    # 目标输出: [target_root_pos(3), target_root_quat_wxyz(4), target_joints...]
    # 约束：保持 pelvis 在世界系下与原始数据一致，再由 target URDF 的 root->pelvis 链反解 root。
    source_joint_index = {name: i for i, name in enumerate(source_joint_names)}

    # 目标 joint_pos 使用 target_joint_names 顺序
    joint_pos = np.zeros((T, len(target_joint_names)), dtype=np.float32)
    base_pos_w = np.zeros((T, 3), dtype=np.float32)
    base_quat_w = np.zeros((T, 4), dtype=np.float32)

    waist_sign_overrides = {
        "waist_yaw_joint": -1.0,
        "waist_roll_joint": -1.0,
        "waist_pitch_joint": -1.0,
    }

    for t in range(T):
        source_pelvis_pos = qpos[t, :3].astype(np.float64)
        source_pelvis_quat = qpos[t, 3:7].astype(np.float64)

        # 构建目标 joint 向量
        target_joint_by_name = {}
        for j, target_joint_name in enumerate(target_joint_names):
            if target_joint_name not in source_joint_index:
                raise ValueError(f"目标关节 '{target_joint_name}' 在源模型 joint_names 中不存在。")
            src_val = float(qpos[t, 7 + source_joint_index[target_joint_name]])
            target_val = src_val * waist_sign_overrides.get(target_joint_name, 1.0)
            joint_pos[t, j] = target_val
            target_joint_by_name[target_joint_name] = target_val

        # 计算目标模型下 root->pelvis 变换
        target_root_to_pelvis_pos, target_root_to_pelvis_quat = compute_root_to_pelvis_transform(
            root_to_pelvis_chain,
            target_joint_by_name,
        )

        # 反解目标 root 世界位姿，使目标 pelvis 与源 pelvis 世界位姿完全一致
        pelvis_to_target_root_pos, pelvis_to_target_root_quat = invert_transform(
            target_root_to_pelvis_pos,
            target_root_to_pelvis_quat,
        )
        target_root_world_pos, target_root_world_quat = compose_transform(
            source_pelvis_pos,
            source_pelvis_quat,
            pelvis_to_target_root_pos,
            pelvis_to_target_root_quat,
        )

        base_pos_w[t] = target_root_world_pos.astype(np.float32)
        base_quat_w[t] = target_root_world_quat.astype(np.float32)

    # ------------------------------------------------------------------
    # 地形缩放校正（terrain_scale != 1.0 时生效）
    # 使用逐帧足底 Z 近似地形高度，比固定 h_standing 更准确。
    # ------------------------------------------------------------------
    if abs(terrain_scale - 1.0) > 1e-6:
        print(f"  [地形校正] terrain_scale = {terrain_scale:.6f}")

        # 逐帧 FK 提取足底 Z
        source_data = mujoco.MjData(source_robot)
        l_ankle_id = mujoco.mj_name2id(source_robot, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        r_ankle_id = mujoco.mj_name2id(source_robot, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

        foot_z = np.zeros(T, dtype=np.float32)
        for t in range(T):
            source_data.qpos[:] = qpos[t]
            mujoco.mj_forward(source_robot, source_data)
            l_foot_z = float(source_data.xpos[l_ankle_id, 2])
            r_foot_z = float(source_data.xpos[r_ankle_id, 2])
            foot_z[t] = min(l_foot_z, r_foot_z)

        # 校正公式:
        #   terrain_z ≈ foot_z  (支撑脚近似地形面)
        #   height_above_terrain = base_z - foot_z  (机器人自身高度，不受地形缩放影响)
        #   corrected_z = foot_z / terrain_scale + height_above_terrain
        #               = foot_z * (1/terrain_scale - 1) + base_z
        raw_z = base_pos_w[:, 2].copy()
        base_pos_w[:, 2] = foot_z * (1.0 / terrain_scale - 1.0) + raw_z

        z_correction = base_pos_w[:, 2] - raw_z
        print(f"  [地形校正] foot_z 范围: [{foot_z.min():.4f}, {foot_z.max():.4f}] m")
        print(f"  [地形校正] z 修正范围: [{z_correction.min():.4f}, {z_correction.max():.4f}] m")
        print(f"  [地形校正] 修正后 base_pos_w z 范围: [{base_pos_w[:, 2].min():.4f}, {base_pos_w[:, 2].max():.4f}]")

    np.savez(
        output_path,
        framerate=np.float32(fps),          # instinct 读取时用 .item()，需是标量
        joint_names=np.array(target_joint_names),  # (DOF,)
        joint_pos=joint_pos,                # (T, DOF)
        base_pos_w=base_pos_w,              # (T, 3)
        base_quat_w=base_quat_w,            # (T, 4) wxyz
    )
    print(f"  已保存到: {output_path}")
    print(f"  帧数: {T},  关节数: {len(target_joint_names)},  帧率: {fps}")
    print("转换完成！")
    print()
    print("注意: 输出文件名必须以 'retargetted.npz' 或 'retargeted.npz' 结尾，")
    print("      instinct 才能正确识别并走 _read_retargetted_motion_file 读取路径。")


def main():
    # 获取项目根目录 (假设脚本在 holosoma/src/holosoma_retargeting/holosoma_retargeting/ 目录下)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[4]
    
    script_dir = script_path.parent
    default_source_xml = script_dir / "models/g1/g1_29dof.xml"
    default_target_urdf = script_dir / "models/g1/g1_29dof_torsobase_popsicle.urdf"
    
    default_input = project_root / "data"
    default_output = project_root / "data_instinct"

    parser = argparse.ArgumentParser(
        description="将 holosoma/omniretarget 原版 npz 转换为 instinct 训练所需的 retargeted 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 批量处理默认文件夹 (data -> data_instinct)，含地形校正
  python omniretargetTOinstinct.py --terrain-scale 0.7415730337078652

  # 指定输入文件夹
  python omniretargetTOinstinct.py --input ./my_data --output ./my_output

  # 单个文件处理（不需要地形校正时可省略 --terrain-scale）
  python omniretargetTOinstinct.py \\
      --input  ../../../../data/stairs_27_original.npz \\
      --output ../../../../data/stairs27_retargeted.npz

  # 带地形校正的单文件处理（mocap_climb_seq_8 的缩放因子）
  python omniretargetTOinstinct.py \\
      --input  ../../../../data/stairs_01_original.npz \\
      --output ../../../../data_instinct/stairs_01_original_retargeted.npz \\
      --terrain-scale 0.7415730337078652

默认 source XML 路径: {default_source_xml}
默认 target URDF 路径: {default_target_urdf}
        """
    )
    parser.add_argument("--input",  default=str(default_input), help=f"输入路径 (可以是单个 npz 文件或文件夹，默认: {default_input})")
    parser.add_argument("--output", default=str(default_output), help=f"输出路径 (可以是文件路径或文件夹，默认: {default_output})")
    parser.add_argument("--source-xml", "--xml", dest="source_xml", default=str(default_source_xml), help=f"源 MuJoCo XML 模型文件路径 (默认: {default_source_xml})")
    parser.add_argument("--target-urdf", dest="target_urdf", default=str(default_target_urdf), help=f"目标 URDF 模型文件路径 (默认: {default_target_urdf})")
    parser.add_argument("--terrain-scale", type=float, default=1.0,
                        help="holosoma retarget 时地形 mesh 的统一缩放因子 (默认 1.0 = 不校正)。"
                             " 例如 mocap_climb_seq_8 数据的缩放因子为 0.7415730337078652"
                             " (见 box_assets.xml)。设置后会按公式修正 base_pos_w 的 z 分量，"
                             "补偿 holosoma 缩放地形与 instinct 原始地形之间的高度差异。")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    source_xml_path = Path(args.source_xml)
    target_urdf_path = Path(args.target_urdf)
    terrain_scale = args.terrain_scale

    if not source_xml_path.exists():
        print(f"错误: 源 XML 模型文件不存在: {source_xml_path}")
        return

    if not target_urdf_path.exists():
        print(f"错误: 目标 URDF 模型文件不存在: {target_urdf_path}")
        return

    if input_path.is_dir():
        # 批量处理模式
        print(f"进入批量处理模式...")
        print(f"输入目录: {input_path}")
        print(f"输出目录: {output_path}")
        
        os.makedirs(output_path, exist_ok=True)
        files = sorted(list(input_path.glob("*.npz")))
        
        # 过滤掉已经是 retargeted 的文件
        process_files = [f for f in files if not (f.name.endswith("retargeted.npz") or f.name.endswith("retargetted.npz"))]
        
        if not process_files:
            print(f"在 {input_path} 中未找到需要处理的 .npz 文件（已自动跳过以 retargeted.npz 结尾的文件）。")
            return

        print(f"找到 {len(process_files)} 个待处理文件，准备开始批量转换...\n")
        for f in process_files:
            # 统一采用：原文件名(stem) + _retargeted.npz
            # 这样可以完美区分 original 和 augmented，且符合 instinct 命名要求
            out_name = f.stem + "_retargeted.npz"
            
            target_out = output_path / out_name
            convert_format(str(f), str(target_out), str(source_xml_path), str(target_urdf_path),
                           terrain_scale=terrain_scale)
        
        print(f"所有文件处理完成！共处理 {len(process_files)} 个文件。")
    
    else:
        # 单文件处理模式
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {input_path}")
            return
            
        # 如果 output 是个目录，则在目录下生成对应的文件名
        final_output = output_path
        if not str(output_path).endswith(".npz"):
            os.makedirs(output_path, exist_ok=True)
            out_name = input_path.stem + "_retargeted.npz"
            final_output = output_path / out_name
        else:
            os.makedirs(output_path.parent, exist_ok=True)

        if not (str(final_output).endswith("retargeted.npz") or str(final_output).endswith("retargetted.npz")):
            print("警告: 输出文件名不以 'retargeted.npz' 或 'retargetted.npz' 结尾，")
            print("      instinct 可能无法正确识别文件类型！")

        convert_format(str(input_path), str(final_output), str(source_xml_path), str(target_urdf_path),
                       terrain_scale=terrain_scale)


if __name__ == "__main__":
    main()
