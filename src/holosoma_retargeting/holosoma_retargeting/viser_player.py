#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402

'''
python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/viser_player.py \
  --mjcf_path /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/mocap_climb_seq_8/g1_29dof_w_multi_boxes.xml \
  --qpos_npz /home/huangyucheng/桌面/Omniretarget/data/stairs_01_original.npz \
  --print_foot_pos \
  --print_interval 20

python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/viser_player.py \
  --mjcf_path /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/mocap_climb_seq_8/g1_29dof_w_multi_boxes.xml \
  --qpos_npz /home/huangyucheng/桌面/Omniretarget/data/




python /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/viser_player.py \
  --mjcf_path /home/huangyucheng/桌面/Omniretarget/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/mocap_climb_seq_8/g1_29dof_w_multi_boxes_scaled_0.74_0.74_0.74.xml \
  --qpos_npz /home/huangyucheng/桌面/Omniretarget/data/stairs_111_original.npz
  
'''


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # expected: qpos [T, ?], and optional fps
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def mat2quat(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    tr = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (mat[2, 1] - mat[1, 2]) / s
        qy = (mat[0, 2] - mat[2, 0]) / s
        qz = (mat[1, 0] - mat[0, 1]) / s
    elif (mat[0, 0] > mat[1, 1]) and (mat[0, 0] > mat[2, 2]):
        s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
        qw = (mat[2, 1] - mat[1, 2]) / s
        qx = 0.25 * s
        qy = (mat[0, 1] + mat[1, 0]) / s
        qz = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
        qw = (mat[0, 2] - mat[2, 0]) / s
        qx = (mat[0, 1] + mat[1, 0]) / s
        qy = 0.25 * s
        qz = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
        qw = (mat[1, 0] - mat[0, 1]) / s
        qx = (mat[0, 2] + mat[2, 0]) / s
        qy = (mat[1, 2] + mat[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    fps: int | None = None,
    npz_files: list[str] | None = None,
    initial_index: int = 0,
):
    server = viser.ViserServer(port=8081)
    actual_fps = fps if fps is not None else config.fps
    n_frames = len(qpos)

    if config.mjcf_path is not None:
        print(f"[viser_player] Loading MJCF: {config.mjcf_path}")
        model = mujoco.MjModel.from_xml_path(config.mjcf_path)
        data = mujoco.MjData(model)
        
        # Create viser nodes for all geoms
        viser_geoms = {}
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            
            g_type = model.geom_type[i]
            # Priority: 1. material rgba, 2. geom rgba
            mat_id = model.geom_matid[i]
            if mat_id != -1:
                rgba = model.mat_rgba[mat_id]
            else:
                rgba = model.geom_rgba[i]
            
            size = model.geom_size[i]
            
            if g_type == mujoco.mjtGeom.mjGEOM_BOX:
                viser_geoms[name] = server.scene.add_box(
                    f"/mjcf/{name}",
                    dimensions=tuple(size * 2),
                    color=rgba[:3]
                )
            elif g_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                viser_geoms[name] = server.scene.add_cylinder(
                    f"/mjcf/{name}",
                    radius=size[0],
                    height=size[1] * 2,
                    color=rgba[:3]
                )
            elif g_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                viser_geoms[name] = server.scene.add_icosphere(
                    f"/mjcf/{name}",
                    radius=size[0],
                    color=rgba[:3]
                )
            elif g_type == mujoco.mjtGeom.mjGEOM_PLANE:
                # MuJoCo plane size is [width, height, spacing]
                viser_geoms[name] = server.scene.add_box(
                    f"/mjcf/{name}",
                    dimensions=(size[0] * 2, size[1] * 2, 0.01),
                    color=rgba[:3]
                )
            elif g_type == mujoco.mjtGeom.mjGEOM_MESH:
                # Load mesh from MuJoCo
                mesh_id = model.geom_dataid[i]
                v_start = model.mesh_vertadr[mesh_id]
                v_num = model.mesh_vertnum[mesh_id]
                f_start = model.mesh_faceadr[mesh_id]
                f_num = model.mesh_facenum[mesh_id]
                
                verts = model.mesh_vert[v_start : v_start + v_num]
                faces = model.mesh_face[f_start : f_start + f_num]
                
                viser_geoms[name] = server.scene.add_mesh_simple(
                    f"/mjcf/{name}",
                    vertices=verts,
                    faces=faces,
                    color=rgba[:3]
                )

        # Add a default grid for reference
        server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height)

        #print foot 
        # Pre-compute specific ankle roll body indices for periodic printing
        foot_body_ids: list[int] = []
        foot_body_names: list[str] = []
        try:
            target_names = {"left_ankle_roll_link", "right_ankle_roll_link"}
            for i in range(model.nbody):
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                if bname in target_names:
                    foot_body_ids.append(i)
                    foot_body_names.append(bname)
            # Fallback: match case-insensitively if exact names not found
            if len(foot_body_ids) < 2:
                foot_body_ids = []
                foot_body_names = []
                for i in range(model.nbody):
                    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                    lname = bname.lower()
                    if lname == "left_ankle_roll_link" or lname == "right_ankle_roll_link":
                        foot_body_ids.append(i)
                        foot_body_names.append(bname)
        except Exception:
            foot_body_ids = []
            foot_body_names = []

        # Sync loop and GUI state
        state = {
            "playing": False,
            "frame": 0,
            "n_frames": n_frames,
            "actual_fps": float(actual_fps),
            "seq_files": (npz_files or []),
            "seq_idx": int(max(0, min(initial_index, (len(npz_files or []) - 1)))) if (npz_files and len(npz_files) > 0) else 0,
            "last_printed_frame": -1, #print foot
        }

        def clamp_frame(val: int) -> int:
            if state["n_frames"] <= 0:
                return 0
            return int(max(0, min(val, state["n_frames"] - 1)))

        def update_viser_from_mj(frame_idx):
            idx = clamp_frame(frame_idx)
            data.qpos[:] = qpos[idx]
            mujoco.mj_forward(model, data)
            for i in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                if name in viser_geoms:
                    node = viser_geoms[name]
                    node.position = data.geom_xpos[i]
                    node.wxyz = mat2quat(data.geom_xmat[i].reshape(3, 3))

            #print foot 
            # Optional: print foot positions at interval
            if config.print_foot_pos and state["n_frames"] > 0:
                if idx != state["last_printed_frame"] and (idx % max(1, int(config.print_interval)) == 0):
                    state["last_printed_frame"] = idx
                    if foot_body_ids:
                        print(f"--- Frame {idx} ---")
                        for bid, bname in zip(foot_body_ids, foot_body_names):
                            pos = data.xpos[bid]  # body world position (3,)
                            if pos is not None:
                                print(f"Body: {bname}, Pos: {np.array2string(pos, precision=4)}")

        # Frame slider with a generous max; we clamp internally to current sequence length.
        frame_slider = server.gui.add_slider("Frame", min=0, max=100000, step=1, initial_value=0)

        @frame_slider.on_update
        def _(event):
            state["frame"] = clamp_frame(event.target.value)
            update_viser_from_mj(state["frame"])

        play_button = server.gui.add_button("Play / Pause")
        @play_button.on_click
        def _(_):
            state["playing"] = not state["playing"]

        # Batch playback controls (if multiple sequences provided)
        if state["seq_files"]:
            current_file_text = server.gui.add_text(
                "Current Sequence",
                initial_value=f"{Path(state['seq_files'][state['seq_idx']]).name}",
            )
            # Use a generous max so changes in filtered list size won't break the UI.
            seq_slider = server.gui.add_slider("Sequence Index", min=0, max=100000, step=1, initial_value=state["seq_idx"])
            prev_btn = server.gui.add_button("Prev Sequence")
            next_btn = server.gui.add_button("Next Sequence")

            # Prepare filter options based on available files
            all_files = list(state["seq_files"])
            has_original = any(p.endswith("_original.npz") for p in all_files)
            has_augmented = any(p.endswith("_augmented.npz") for p in all_files)
            filter_options = []
            if has_original:
                filter_options.append("original")
            if has_augmented:
                filter_options.append("augmented")
            filter_options.insert(0, "all")  # "all" first
            state["filter_mode"] = "all"

            def apply_filter():
                mode = state.get("filter_mode", "all")
                if mode == "original":
                    return [p for p in all_files if p.endswith("_original.npz")]
                if mode == "augmented":
                    return [p for p in all_files if p.endswith("_augmented.npz")]
                return list(all_files)

            # Helper to refresh filtered list and clamp index
            def refresh_filtered_and_clamp():
                state["seq_files"] = apply_filter()
                if not state["seq_files"]:
                    return False
                # Re-map by filename if possible to preserve current file selection
                try:
                    current_name = Path(all_files[state["seq_idx"]]).name
                    names = [Path(p).name for p in state["seq_files"]]
                    if current_name in names:
                        state["seq_idx"] = names.index(current_name)
                    else:
                        state["seq_idx"] = 0
                except Exception:
                    state["seq_idx"] = 0
                try:
                    seq_slider.value = state["seq_idx"]
                except Exception:
                    pass
                try:
                    current_file_text.value = f"{Path(state['seq_files'][state['seq_idx']]).name}"
                except Exception:
                    pass
                return True

            # Dropdown to select filter mode
            filter_dropdown = server.gui.add_dropdown(
                "Sequence Filter",
                options=filter_options,
                initial_value=state["filter_mode"],
            )

            @filter_dropdown.on_update
            def _(event):
                state["filter_mode"] = str(event.target.value)
                ok = refresh_filtered_and_clamp()
                if ok:
                    # Load the currently selected file after filter change
                    npz_path = state["seq_files"][state["seq_idx"]]
                    try:
                        new_qpos, new_fps = load_npz(npz_path)
                    except Exception as e:
                        print(f"[viser_player] Failed to load {npz_path}: {e}")
                        return
                    nonlocal qpos, actual_fps
                    qpos = new_qpos
                    actual_fps = float(new_fps if new_fps is not None else config.fps)
                    state["n_frames"] = len(qpos)
                    state["frame"] = 0
                    try:
                        frame_slider.value = 0
                    except Exception:
                        pass
                    update_viser_from_mj(0)

            # Also provide a filename dropdown to jump directly
            def get_current_names():
                return [Path(p).name for p in state["seq_files"]]

            filename_dropdown = server.gui.add_dropdown(
                "Sequence File",
                options=get_current_names(),
                initial_value=Path(state["seq_files"][state["seq_idx"]]).name,
            )

            @filename_dropdown.on_update
            def _(event):
                name = str(event.target.value)
                names = get_current_names()
                if name in names:
                    idx = names.index(name)
                    switch_sequence(idx)

            def switch_sequence(new_idx: int):
                nonlocal qpos, actual_fps
                if not state["seq_files"]:
                    return
                clamped = int(max(0, min(new_idx, len(state["seq_files"]) - 1)))
                if clamped == state["seq_idx"]:
                    return
                state["seq_idx"] = clamped
                npz_path = state["seq_files"][state["seq_idx"]]
                try:
                    new_qpos, new_fps = load_npz(npz_path)
                except Exception as e:
                    print(f"[viser_player] Failed to load {npz_path}: {e}")
                    return
                qpos = new_qpos
                actual_fps = float(new_fps if new_fps is not None else config.fps)
                state["n_frames"] = len(qpos)
                state["frame"] = 0
                try:
                    current_file_text.value = f"{Path(npz_path).name}"
                except Exception:
                    pass
                try:
                    seq_slider.value = state["seq_idx"]
                except Exception:
                    pass
                try:
                    # Update filename dropdown options and selection
                    filename_dropdown.options = get_current_names()
                    filename_dropdown.value = Path(npz_path).name
                except Exception:
                    pass
                try:
                    frame_slider.value = 0
                except Exception:
                    pass
                update_viser_from_mj(0)

            @seq_slider.on_update
            def _(event):
                # Clamp to available filtered list
                if not state["seq_files"]:
                    return
                idx = int(max(0, min(int(event.target.value), len(state["seq_files"]) - 1)))
                switch_sequence(idx)

            @prev_btn.on_click
            def _(_):
                if not state["seq_files"]:
                    return
                new_idx = (state["seq_idx"] - 1) % len(state["seq_files"])
                switch_sequence(new_idx)

            @next_btn.on_click
            def _(_):
                if not state["seq_files"]:
                    return
                new_idx = (state["seq_idx"] + 1) % len(state["seq_files"])
                switch_sequence(new_idx)

        def playback_thread():
            while True:
                if state["playing"]:
                    if state["n_frames"] > 0:
                        state["frame"] = (state["frame"] + 1) % state["n_frames"]
                        update_viser_from_mj(state["frame"])
                    time.sleep(1.0 / max(1e-6, state["actual_fps"]))
                else:
                    time.sleep(0.1)

        threading.Thread(target=playback_thread, daemon=True).start()
        update_viser_from_mj(0)

    else:
        # Standard URDF player
        robot_root = server.scene.add_frame("/robot", show_axes=False)
        robot_urdf_y = yourdfpy.URDF.load(config.robot_urdf, load_meshes=True, build_scene_graph=True)
        vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

        vo = None
        if config.object_urdf:
            object_root = server.scene.add_frame("/object", show_axes=False)
            object_urdf_y = yourdfpy.URDF.load(config.object_urdf, load_meshes=True, build_scene_graph=True)
            vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

        server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height)
        robot_dof = len(vr.get_actuated_joint_limits())

        create_motion_control_sliders(
            server=server,
            viser_robot=vr,
            robot_base_frame=robot_root,
            motion_sequence=qpos,
            robot_dof=robot_dof,
            viser_object=vo if config.assume_object_in_qpos else None,
            object_base_frame=None if not config.assume_object_in_qpos else server.scene.add_frame("/object_ref"),
            initial_fps=actual_fps,
            loop=config.loop,
        )

    print(f"[viser_player] Running at {server.get_host()}")
    return server


def main(cfg: ViserConfig) -> None:
    # Batch support: directory or glob pattern loads multiple .npz files
    qpos_path = Path(cfg.qpos_npz)
    npz_files: list[str] = []
    initial_index = 0
    if qpos_path.is_dir():
        npz_files = sorted([str(p) for p in qpos_path.glob("*.npz")])
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in directory: {qpos_path}")
        qpos, fps = load_npz(npz_files[initial_index])
    else:
        if any(ch in cfg.qpos_npz for ch in ["*", "?", "["]):
            globbed = sorted([str(p) for p in Path().glob(cfg.qpos_npz)])
            if not globbed:
                raise FileNotFoundError(f"No files matched pattern: {cfg.qpos_npz}")
            npz_files = globbed
            qpos, fps = load_npz(npz_files[initial_index])
        else:
            qpos, fps = load_npz(cfg.qpos_npz)
            npz_files = []
            initial_index = 0
    make_player(config=cfg, qpos=qpos, fps=fps, npz_files=npz_files, initial_index=initial_index)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
