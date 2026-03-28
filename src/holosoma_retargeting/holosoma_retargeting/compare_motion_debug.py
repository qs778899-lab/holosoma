import argparse
import contextlib
import os
from pathlib import Path
from typing import List, Tuple

import mujoco
import numpy as np


@contextlib.contextmanager
def cd(newdir: str):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def load_model(xml_path: str) -> mujoco.MjModel:
    xml = Path(xml_path).resolve()
    with cd(str(xml.parent)):
        return mujoco.MjModel.from_xml_path(xml.name)


def get_actuated_joint_names(model: mujoco.MjModel) -> List[str]:
    names: List[str] = []
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            names.append(model.joint(i).name)
    return names


def build_qpos_from_converted(
    converted_npz: str,
    model: mujoco.MjModel,
) -> Tuple[np.ndarray, float]:
    data = np.load(converted_npz, allow_pickle=True)
    framerate = float(np.array(data["framerate"]).reshape(-1)[0])
    base_pos = np.array(data["base_pos_w"], dtype=np.float64)
    base_quat = np.array(data["base_quat_w"], dtype=np.float64)
    joint_pos = np.array(data["joint_pos"], dtype=np.float64)
    joint_names = data["joint_names"].tolist()

    qpos = np.zeros((joint_pos.shape[0], model.nq), dtype=np.float64)
    qpos[:, :3] = base_pos
    qpos[:, 3:7] = base_quat

    model_joints = get_actuated_joint_names(model)
    idx_map = []
    for jn in model_joints:
        if jn not in joint_names:
            raise ValueError(f"Joint '{jn}' not found in converted data joint_names.")
        idx_map.append(joint_names.index(jn))
    qpos[:, 7:] = joint_pos[:, idx_map]

    return qpos, framerate


def build_qpos_from_original(original_npz: str) -> Tuple[np.ndarray, float]:
    data = np.load(original_npz, allow_pickle=True)
    qpos = np.array(data["qpos"], dtype=np.float64)
    fps_raw = data.get("fps", 30.0)
    fps = float(fps_raw) if not isinstance(fps_raw, np.ndarray) else float(np.array(fps_raw).reshape(-1)[0])
    return qpos, fps


def fk_body_positions_z(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,
    left_link: str,
    right_link: str,
    base_body: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_link)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_link)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)

    if left_id < 0 or right_id < 0 or base_id < 0:
        raise ValueError(
            f"Body id lookup failed. left={left_link}:{left_id}, right={right_link}:{right_id}, base={base_body}:{base_id}"
        )

    left_z = np.zeros(qpos_seq.shape[0], dtype=np.float64)
    right_z = np.zeros(qpos_seq.shape[0], dtype=np.float64)
    base_z = np.zeros(qpos_seq.shape[0], dtype=np.float64)

    for i in range(qpos_seq.shape[0]):
        data.qpos[:] = qpos_seq[i]
        mujoco.mj_forward(model, data)
        left_z[i] = data.xpos[left_id, 2]
        right_z[i] = data.xpos[right_id, 2]
        base_z[i] = data.xpos[base_id, 2]

    return left_z, right_z, base_z


def resample_to_common_time(
    sig_a: np.ndarray,
    fps_a: float,
    sig_b: np.ndarray,
    fps_b: float,
    target_fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_a = np.arange(len(sig_a), dtype=np.float64) / fps_a
    t_b = np.arange(len(sig_b), dtype=np.float64) / fps_b
    t_end = min(t_a[-1], t_b[-1])
    t = np.arange(0.0, t_end, 1.0 / target_fps, dtype=np.float64)
    a_rs = np.interp(t, t_a, sig_a)
    b_rs = np.interp(t, t_b, sig_b)
    return t, a_rs, b_rs


def compare_signal(name: str, t: np.ndarray, a: np.ndarray, b: np.ndarray):
    diff = b - a
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    corr = np.corrcoef(a, b)[0, 1] if len(a) > 1 else np.nan
    print(f"[{name}] MAE={mae:.6f}, RMSE={rmse:.6f}, Corr={corr:.6f}")
    for sec in [0.0, 0.5, 1.0, 2.0, 3.0]:
        if len(t) == 0:
            break
        idx = np.argmin(np.abs(t - sec))
        print(f"  t={t[idx]:.2f}s | orig={a[idx]:.4f} | conv={b[idx]:.4f} | diff={diff[idx]:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare ankle Z trajectories between original and converted motions.")
    parser.add_argument("--original", required=True, help="Path to original holosoma npz (contains qpos/fps).")
    parser.add_argument("--converted", required=True, help="Path to converted retargeted npz.")
    parser.add_argument("--orig-xml", required=True, help="MuJoCo XML used for original qpos semantics.")
    parser.add_argument(
        "--conv-xml",
        default=None,
        help="MuJoCo XML for converted semantics. If not set, use --orig-xml.",
    )
    parser.add_argument("--orig-base-body", default="pelvis", help="Base body name in original model.")
    parser.add_argument("--conv-base-body", default="torso_link", help="Base body name in converted model.")
    parser.add_argument("--left-link", default="left_ankle_roll_link", help="Left ankle body name.")
    parser.add_argument("--right-link", default="right_ankle_roll_link", help="Right ankle body name.")
    parser.add_argument("--target-fps", type=float, default=100.0, help="Common resample fps for fair comparison.")
    args = parser.parse_args()

    conv_xml = args.conv_xml if args.conv_xml is not None else args.orig_xml

    print("Loading models...")
    orig_model = load_model(args.orig_xml)
    conv_model = load_model(conv_xml)

    print("Loading motions...")
    qpos_orig, fps_orig = build_qpos_from_original(args.original)
    qpos_conv, fps_conv = build_qpos_from_converted(args.converted, conv_model)
    print(f"  original: frames={len(qpos_orig)}, fps={fps_orig}")
    print(f"  converted: frames={len(qpos_conv)}, fps={fps_conv}")

    print("Running FK...")
    lz_o, rz_o, bz_o = fk_body_positions_z(
        orig_model, qpos_orig, args.left_link, args.right_link, args.orig_base_body
    )
    lz_c, rz_c, bz_c = fk_body_positions_z(
        conv_model, qpos_conv, args.left_link, args.right_link, args.conv_base_body
    )

    # world z compare
    t, lz_o_rs, lz_c_rs = resample_to_common_time(lz_o, fps_orig, lz_c, fps_conv, args.target_fps)
    _, rz_o_rs, rz_c_rs = resample_to_common_time(rz_o, fps_orig, rz_c, fps_conv, args.target_fps)

    # relative z compare (ankle z - base z)
    lrel_o = lz_o - bz_o
    rrel_o = rz_o - bz_o
    lrel_c = lz_c - bz_c
    rrel_c = rz_c - bz_c
    _, lrel_o_rs, lrel_c_rs = resample_to_common_time(lrel_o, fps_orig, lrel_c, fps_conv, args.target_fps)
    _, rrel_o_rs, rrel_c_rs = resample_to_common_time(rrel_o, fps_orig, rrel_c, fps_conv, args.target_fps)

    print("\n=== World Z Compare ===")
    compare_signal("left_ankle_world_z", t, lz_o_rs, lz_c_rs)
    compare_signal("right_ankle_world_z", t, rz_o_rs, rz_c_rs)

    print("\n=== Relative Z Compare (ankle_z - base_z) ===")
    compare_signal("left_ankle_rel_z", t, lrel_o_rs, lrel_c_rs)
    compare_signal("right_ankle_rel_z", t, rrel_o_rs, rrel_c_rs)

    print("\nDone.")


if __name__ == "__main__":
    main()
