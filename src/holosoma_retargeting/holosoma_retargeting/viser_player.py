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
):
    server = viser.ViserServer()
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
            if name == "floor":
                server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height)
                continue
                
            g_type = model.geom_type[i]
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

        # Sync loop
        state = {"playing": False, "frame": 0}

        @server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0).on_update
        def _(event):
            state["frame"] = event.target.value
            update_viser_from_mj(state["frame"])

        def update_viser_from_mj(frame_idx):
            data.qpos[:] = qpos[frame_idx]
            mujoco.mj_forward(model, data)
            for i in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                if name in viser_geoms:
                    node = viser_geoms[name]
                    node.position = data.geom_xpos[i]
                    node.wxyz = mat2quat(data.geom_xmat[i].reshape(3, 3))

        play_button = server.gui.add_button("Play / Pause")
        @play_button.on_click
        def _(_):
            state["playing"] = not state["playing"]

        def playback_thread():
            while True:
                if state["playing"]:
                    state["frame"] = (state["frame"] + 1) % n_frames
                    update_viser_from_mj(state["frame"])
                    time.sleep(1.0 / actual_fps)
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
        
        from holosoma_retargeting.src.viser_utils import create_motion_control_sliders
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
    qpos, fps = load_npz(cfg.qpos_npz)
    make_player(config=cfg, qpos=qpos, fps=fps)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
