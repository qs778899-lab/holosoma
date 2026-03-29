import numpy as np
import argparse
import mujoco
import os
import contextlib
from pathlib import Path

'''
python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/omniretargetTOinstinct.py \
    --input /home/huangyucheng/桌面/Omniretarget/data/stairs_01_original.npz \
    --output /home/huangyucheng/桌面/Omniretarget/data_instinct/stairs_01_original_retargeted.npz

python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/omniretargetTOinstinct.py \
    --input /home/huangyucheng/桌面/Omniretarget/data \
    --output /home/huangyucheng/桌面/Omniretarget/data_instinct

说明：
  holosoma 原版 npz 格式 (qpos):
      shape = (T, 7 + DOF)
      列顺序 = [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, joint_0, ..., joint_N]
      root = pelvis（标准 MuJoCo qpos 格式）

  instinct 训练所需 retargeted npz 格式（由 amass_motion.py _read_retargetted_motion_file 读取）:
      framerate   scalar        — 帧率（注意：是 framerate，不是 fps）
      joint_names (DOF,)        — 关节名称（不含 freejoint，与 MuJoCo XML 关节顺序一致）
      joint_pos   (T, DOF)      — 纯关节角度（不含 base 的 7 列）
      base_pos_w  (T, 3)        — root 位置（世界坐标，pelvis）
      base_quat_w (T, 4)        — root 四元数 wxyz（世界坐标，pelvis）

  注意：
  - instinct 项目需要配合使用 g1_29dof_pelvisbase_simplified.urdf（以 pelvis 为根的 URDF），
    路径: InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_pelvisbase_simplified.urdf
    这样 instinct 的 pytorch_kinematics FK 与 holosoma MuJoCo FK 完全一致，ankle z 误差为 0。
  - 文件名必须以 "retargetted.npz" 或 "retargeted.npz" 结尾，
    instinct 才会走 _read_retargetted_motion_file 这个读取路径。
  - 不再需要对 waist 关节取反，也不需要切换 base 到 torso_link。
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


def convert_format(input_path: str, output_path: str, xml_path: str):
    """
    将 holosoma 原版 omniretarget npz 转换为 instinct 训练所需的 retargeted 格式。

    holosoma 原版格式:
        qpos: (T, 7+DOF)  标准 MuJoCo qpos [pos(3), quat_wxyz(4), joints(DOF)]
                          root = pelvis
        fps:  scalar

    instinct retargeted 格式（_read_retargetted_motion_file 读取）:
        framerate:   scalar
        joint_names: (DOF,)    不含 freejoint，顺序与 MuJoCo XML 一致
        joint_pos:   (T, DOF)  纯关节角，不含 base 的 7 列，直接使用原始值（不取反）
        base_pos_w:  (T, 3)    pelvis 世界位置，直接从 qpos[:, :3] 读取
        base_quat_w: (T, 4)    pelvis 世界四元数 wxyz，直接从 qpos[:, 3:7] 读取

    配合使用: instinct 项目的 robot_model_path 需指向 g1_29dof_pelvisbase_simplified.urdf
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
    print(f"[2/3] 加载 MuJoCo XML 模型: {xml_path}")
    xml_dir = str(Path(xml_path).parent)
    xml_name = Path(xml_path).name

    try:
        with cd(xml_dir):
            robot = mujoco.MjModel.from_xml_path(xml_name)
    except Exception as e:
        print(f"错误: 加载 XML 失败: {e}")
        return

    print(f"  模型 nq={robot.nq}, njnt={robot.njnt}")

    if robot.nq != nq:
        print(f"警告: XML 模型 nq={robot.nq} 与 qpos 列数 {nq} 不一致！请确认使用了正确的模型文件。")

    # 提取关节名称（跳过 freejoint），顺序与 MuJoCo qpos[7:] 严格对应
    joint_names = []
    for i in range(robot.njnt):
        if robot.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            joint_names.append(robot.joint(i).name)

    dof = len(joint_names)
    print(f"  关节数 (不含 freejoint): {dof}")

    if dof != nq - 7:
        print(f"警告: 关节数 {dof} 与 qpos 关节列数 {nq - 7} 不一致！")

    # ------------------------------------------------------------------
    # 3. 直接拆分 qpos 并保存（无需任何语义转换）
    # ------------------------------------------------------------------
    print(f"[3/3] 拆分数据并保存: {output_path}")

    # MuJoCo qpos 标准布局: [pos(3), quat_wxyz(4), joints(DOF)]
    # root = pelvis，与 g1_29dof_pelvisbase_simplified.urdf 的根节点一致
    base_pos_w  = qpos[:, :3].copy()    # (T, 3)  pelvis 世界位置
    base_quat_w = qpos[:, 3:7].copy()   # (T, 4)  pelvis 世界四元数 wxyz
    joint_pos   = qpos[:, 7:].copy()    # (T, DOF) 纯关节角，直接使用，不取反

    np.savez(
        output_path,
        framerate=np.float32(fps),          # instinct 读取时用 .item()，需是标量
        joint_names=np.array(joint_names),  # (DOF,)
        joint_pos=joint_pos,                # (T, DOF)
        base_pos_w=base_pos_w,              # (T, 3)
        base_quat_w=base_quat_w,            # (T, 4) wxyz
    )
    print(f"  已保存到: {output_path}")
    print(f"  帧数: {T},  关节数: {dof},  帧率: {fps}")
    print("转换完成！")
    print()
    print("重要提示: instinct 项目的 robot_model_path 需指向 g1_29dof_pelvisbase_simplified.urdf")
    print("          文件名必须以 'retargetted.npz' 或 'retargeted.npz' 结尾。")


def main():
    # 获取项目根目录 (假设脚本在 holosoma/src/holosoma_retargeting/holosoma_retargeting/ 目录下)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[4]

    script_dir = script_path.parent
    default_xml = script_dir / "models/g1/g1_29dof.xml"

    default_input = project_root / "data"
    default_output = project_root / "data_instinct"

    parser = argparse.ArgumentParser(
        description="将 holosoma/omniretarget 原版 npz 转换为 instinct 训练所需的 retargeted 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
说明:
  转换后的 npz 文件使用 pelvis 作为 root（base），关节角直接使用，无需取反。
  instinct 项目需配合使用 g1_29dof_pelvisbase_simplified.urdf 作为机器人模型。

示例:
  # 批量处理默认文件夹 (data -> data_instinct)
  python omniretargetTOinstinct.py

  # 指定输入文件夹
  python omniretargetTOinstinct.py --input ./my_data --output ./my_output

  # 单个文件处理
  python omniretargetTOinstinct.py \\
      --input  ../../../../data/stairs_27_original.npz \\
      --output ../../../../data_instinct/stairs_27_original_retargeted.npz

默认 XML 路径: {default_xml}
        """
    )
    parser.add_argument("--input",  default=str(default_input), help=f"输入路径 (可以是单个 npz 文件或文件夹，默认: {default_input})")
    parser.add_argument("--output", default=str(default_output), help=f"输出路径 (可以是文件路径或文件夹，默认: {default_output})")
    parser.add_argument("--xml",    default=str(default_xml), help=f"MuJoCo XML 模型文件路径 (默认: {default_xml})")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    xml_path = Path(args.xml)

    if not xml_path.exists():
        print(f"错误: XML 模型文件不存在: {xml_path}")
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
            out_name = f.stem + "_retargeted.npz"
            target_out = output_path / out_name
            convert_format(str(f), str(target_out), str(xml_path))

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

        convert_format(str(input_path), str(final_output), str(xml_path))


if __name__ == "__main__":
    main()
