from __future__ import annotations

import argparse
import contextlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

'''
python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/omniretargetTOinstinct.py \
    --input /home/huangyucheng/桌面/Omniretarget/data/stairs_01_augmented_interp.npz \
    --output /home/huangyucheng/桌面/Omniretarget/data_instinct/stairs_01_augmented_interp_retargeted.npz 
python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/omniretargetTOinstinct.py \
    --input /home/huangyucheng/桌面/Omniretarget/data/stairs146_augmented_interp.npz \
    --output /home/huangyucheng/桌面/Omniretarget/data_instinct/stairs146_augmented_interp_retargeted.npz 

'''

'''
格式转换原理的关键点解释：

1. omniretarget原始npz的qpos(需要ominretarget使用的机器人模型文件:g1_29dof.xml):
pelvis_pos(3), pelvis_quat_wxyz(4), source_joints

2. instinct目标npz的核心(需要instinct训练使用的机器人模型文件:g1_29dof_torsobase_popsicle.urdf):
torso_pos(3), torso_quat_wxyz(4), target_joints

3. 对 waist 三个关节做符号翻转:
source 模型是 pelvis-root 语义；target 模型是 torsobase 语义。两边 pelvis 和 torso 之间那条 waist 链的父子方向反了(且多个旋转矩阵乘法的位置不可随意交换)。 
因此转换后输出到 Instinct joint_pos 里的这三个腰部关节角(waist_yaw_joint, waist_roll_joint, waist_pitch_joint)数值本身都需要乘-1。

'''



@contextlib.contextmanager
def cd(newdir: str):
    """Temporarily switch working directory for MuJoCo relative asset loading."""
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

    all_links: set[str] = set()
    child_links: set[str] = set()
    actuated_joint_names: list[str] = []
    joint_by_child: dict[str, dict] = {}

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

        xyz = (
            np.fromstring(origin_elem.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
            if origin_elem is not None
            else np.zeros(3, dtype=np.float64)
        )
        rpy = (
            np.fromstring(origin_elem.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
            if origin_elem is not None
            else np.zeros(3, dtype=np.float64)
        )
        axis = (
            np.fromstring(axis_elem.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
            if axis_elem is not None
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

        joint_info = {
            "name": name,
            "type": joint_type,
            "parent": parent_link,
            "child": child_link,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
        }
        if child_link:
            joint_by_child[child_link] = joint_info
        if joint_type not in ("fixed", "floating") and name is not None:
            actuated_joint_names.append(name)

    root_candidates = sorted(all_links - child_links)
    if len(root_candidates) != 1:
        raise ValueError(f"Could not determine a unique root link from target URDF: {root_candidates}")
    root_link = root_candidates[0]

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


def load_source_joint_names(source_xml_path: str, nq_expected: int) -> list[str]:
    """Load source MuJoCo XML and extract actuated joint names matching qpos[7:]."""
    xml_dir = str(Path(source_xml_path).parent)
    xml_name = Path(source_xml_path).name
    with cd(xml_dir):
        source_robot = mujoco.MjModel.from_xml_path(xml_name)

    print(f"  source model nq={source_robot.nq}, njnt={source_robot.njnt}")
    if source_robot.nq != nq_expected:
        print(
            f"警告: 源 XML 模型 nq={source_robot.nq} 与 qpos 列数 {nq_expected} 不一致！"
            " 请确认使用了正确的 source XML。"
        )

    source_joint_names = []
    for i in range(source_robot.njnt):
        if source_robot.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            source_joint_names.append(source_robot.joint(i).name)

    if len(source_joint_names) != nq_expected - 7:
        print(
            f"警告: 源模型关节数 {len(source_joint_names)} 与 qpos 关节列数 {nq_expected - 7} 不一致！"
        )
    return source_joint_names


def validate_target_projection(
    source_pelvis_pos: np.ndarray,
    source_pelvis_quat: np.ndarray,
    root_to_pelvis_pos: np.ndarray,
    root_to_pelvis_quat: np.ndarray,
    target_root_world_pos: np.ndarray,
    target_root_world_quat: np.ndarray,
) -> tuple[float, float]:
    """Check reconstructed target root reproduces the original pelvis world pose."""
    recon_pelvis_pos, recon_pelvis_quat = compose_transform(
        target_root_world_pos,
        target_root_world_quat,
        root_to_pelvis_pos,
        root_to_pelvis_quat,
    )
    pos_err = float(np.linalg.norm(recon_pelvis_pos - source_pelvis_pos))
    dot = abs(float(np.dot(recon_pelvis_quat, source_pelvis_quat)))
    dot = min(1.0, max(-1.0, dot))
    quat_err = float(2.0 * np.arccos(dot))
    return pos_err, quat_err


def convert_format(
    input_path: str,
    output_path: str,
    source_xml_path: str,
    target_urdf_path: str,
):
    """
    Convert holosoma omniretarget output NPZ into Instinct retargeted format.

    Input NPZ (source):
        qpos: (T, 7 + DOF) = [pelvis_pos(3), pelvis_quat_wxyz(4), source_joints...]
        fps: scalar

    Output NPZ (target):
        framerate: scalar
        joint_names: (DOF,) target URDF actuated joint order
        joint_pos: (T, DOF)
        base_pos_w: (T, 3) target root position in world
        base_quat_w: (T, 4) target root quaternion wxyz in world
    """
    print(f"[1/3] 加载原始 holosoma npz: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    if "qpos" not in data:
        raise ValueError("输入文件中没有 'qpos' 字段，请确认是 holosoma/omniretarget 的输出文件。")

    qpos = data["qpos"].astype(np.float32)
    num_frames, nq = qpos.shape
    fps_raw = data.get("fps", 30.0)
    fps = float(fps_raw) if not isinstance(fps_raw, np.ndarray) else float(fps_raw.flat[0])
    print(f"  qpos shape: {qpos.shape}")
    print(f"  fps: {fps}")

    print(f"[2/3] 解析 source XML 与 target URDF")
    source_joint_names = load_source_joint_names(source_xml_path, nq)
    target_joint_names, root_to_pelvis_chain, target_root_link = parse_target_urdf(target_urdf_path)
    print(f"  target root link: {target_root_link}")
    print(f"  source joints: {len(source_joint_names)}, target joints: {len(target_joint_names)}")

    source_joint_index = {name: i for i, name in enumerate(source_joint_names)}
    missing_joints = [name for name in target_joint_names if name not in source_joint_index]
    if missing_joints:
        raise ValueError(
            "目标 URDF 中以下关节在源 qpos 对应的 source XML 里找不到: "
            f"{missing_joints}"
        )

    print(f"[3/3] 重建 target torsobase root 并保存: {output_path}")
    joint_pos = np.zeros((num_frames, len(target_joint_names)), dtype=np.float32)
    base_pos_w = np.zeros((num_frames, 3), dtype=np.float32)
    base_quat_w = np.zeros((num_frames, 4), dtype=np.float32)

    waist_sign_overrides = {
        "waist_yaw_joint": -1.0,
        "waist_roll_joint": -1.0,
        "waist_pitch_joint": -1.0,
    }

    max_pelvis_pos_err = 0.0
    max_pelvis_quat_err = 0.0

    for frame_idx in range(num_frames):
        source_pelvis_pos = qpos[frame_idx, :3].astype(np.float64)
        source_pelvis_quat = qpos[frame_idx, 3:7].astype(np.float64)

        target_joint_by_name: dict[str, float] = {}
        for j, target_joint_name in enumerate(target_joint_names):
            src_val = float(qpos[frame_idx, 7 + source_joint_index[target_joint_name]])
            target_val = src_val * waist_sign_overrides.get(target_joint_name, 1.0)
            joint_pos[frame_idx, j] = target_val
            target_joint_by_name[target_joint_name] = target_val

        root_to_pelvis_pos, root_to_pelvis_quat = compute_root_to_pelvis_transform(
            root_to_pelvis_chain,
            target_joint_by_name,
        )
        pelvis_to_root_pos, pelvis_to_root_quat = invert_transform(
            root_to_pelvis_pos,
            root_to_pelvis_quat,
        )
        target_root_world_pos, target_root_world_quat = compose_transform(
            source_pelvis_pos,
            source_pelvis_quat,
            pelvis_to_root_pos,
            pelvis_to_root_quat,
        )

        pos_err, quat_err = validate_target_projection(
            source_pelvis_pos,
            source_pelvis_quat,
            root_to_pelvis_pos,
            root_to_pelvis_quat,
            target_root_world_pos,
            target_root_world_quat,
        )
        max_pelvis_pos_err = max(max_pelvis_pos_err, pos_err)
        max_pelvis_quat_err = max(max_pelvis_quat_err, quat_err)

        base_pos_w[frame_idx] = target_root_world_pos.astype(np.float32)
        base_quat_w[frame_idx] = (target_root_world_quat / (np.linalg.norm(target_root_world_quat) + 1e-12)).astype(np.float32)

    np.savez(
        output_path,
        framerate=np.float32(fps),
        joint_names=np.array(target_joint_names),
        joint_pos=joint_pos,
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
    )
    print(f"  已保存到: {output_path}")
    print(f"  帧数: {num_frames}, 关节数: {len(target_joint_names)}, 帧率: {fps}")
    print(f"  pelvis consistency max position error: {max_pelvis_pos_err:.6e} m")
    print(f"  pelvis consistency max orientation error: {np.rad2deg(max_pelvis_quat_err):.6e} deg")
    print("转换完成！")
    print("注意: 输出文件名必须以 'retargetted.npz' 或 'retargeted.npz' 结尾，Instinct 才会按 retargeted motion 读取。")


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[4]
    script_dir = script_path.parent

    default_source_xml = script_dir / "models/g1/g1_29dof.xml"
    default_target_urdf = Path(
        "/home/huangyucheng/桌面/Project Instinct/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf"
    )
    default_input = project_root / "data"
    default_output = project_root / "data_instinct"

    parser = argparse.ArgumentParser(
        description="将 holosoma/omniretarget 原版 npz 转换为 Instinct 训练所需的 retargeted 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  python omniretargetTOinstinct.py \
      --input /path/to/stairs_01_original.npz \
      --output /path/to/stairs_01_original_retargeted.npz

  python omniretargetTOinstinct.py \
      --input /path/to/source_dir \
      --output /path/to/output_dir

默认 source XML: {default_source_xml}
默认 target URDF: {default_target_urdf}
""",
    )
    parser.add_argument("--input", default=str(default_input), help=f"输入路径 (单个 npz 文件或文件夹，默认: {default_input})")
    parser.add_argument("--output", default=str(default_output), help=f"输出路径 (文件路径或文件夹，默认: {default_output})")
    parser.add_argument(
        "--source-xml",
        "--xml",
        dest="source_xml",
        default=str(default_source_xml),
        help=f"解释源 qpos 语义的 MuJoCo XML (默认: {default_source_xml})",
    )
    parser.add_argument(
        "--target-urdf",
        dest="target_urdf",
        default=str(default_target_urdf),
        help=f"Instinct 训练实际使用的 target torsobase URDF (默认: {default_target_urdf})",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    source_xml_path = Path(args.source_xml)
    target_urdf_path = Path(args.target_urdf)

    if not source_xml_path.exists():
        raise FileNotFoundError(f"源 XML 模型文件不存在: {source_xml_path}")
    if not target_urdf_path.exists():
        raise FileNotFoundError(f"目标 URDF 模型文件不存在: {target_urdf_path}")

    if input_path.is_dir():
        print("进入批量处理模式...")
        print(f"输入目录: {input_path}")
        print(f"输出目录: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        files = sorted(list(input_path.glob("*.npz")))
        process_files = [
            f for f in files if not (f.name.endswith("retargeted.npz") or f.name.endswith("retargetted.npz"))
        ]
        if not process_files:
            print(f"在 {input_path} 中未找到需要处理的 .npz 文件（已跳过 retargeted 文件）。")
            return
        print(f"找到 {len(process_files)} 个待处理文件。\n")
        for f in process_files:
            out_name = f.stem + "_retargeted.npz"
            target_out = output_path / out_name
            convert_format(str(f), str(target_out), str(source_xml_path), str(target_urdf_path))
        print(f"所有文件处理完成！共处理 {len(process_files)} 个文件。")
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        final_output = output_path
        if not str(output_path).endswith(".npz"):
            os.makedirs(output_path, exist_ok=True)
            final_output = output_path / f"{input_path.stem}_retargeted.npz"
        else:
            os.makedirs(output_path.parent, exist_ok=True)

        if not (str(final_output).endswith("retargeted.npz") or str(final_output).endswith("retargetted.npz")):
            print("警告: 输出文件名不以 'retargeted.npz' 或 'retargetted.npz' 结尾，Instinct 可能无法按目标路径识别。")

        convert_format(str(input_path), str(final_output), str(source_xml_path), str(target_urdf_path))


if __name__ == "__main__":
    main()
