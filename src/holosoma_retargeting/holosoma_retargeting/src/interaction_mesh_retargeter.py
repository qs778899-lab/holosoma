from __future__ import annotations

import sys
import time
import os
from datetime import datetime
from pathlib import Path
from types import ModuleType

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from tqdm import tqdm
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import with type ignore for mypy compatibility
from mujoco_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    _world_mesh_from_geom,
)
from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    transform_points_local_to_world,
    transform_points_world_to_local,
)
from viser_utils import create_motion_control_sliders  # type: ignore[import-not-found,no-redef]  # noqa: E402


class InteractionMeshRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str,
        # Snooker 相关参数（无默认值，必须显式传入）
        activate_realtime_rotation_tracking: bool,  # 数据源开关：当输入只有.npy 文件，没有参考的*_original.npz文件时，从 7D 数据中实时抠出真人的手腕旋转，强行喂给解算器作为“唯一引导”。
        activate_snooker_tracking: bool,  # 是否启用左手腕 Yaw Tracking
        activate_general_nominal_tracking: bool,  # 是否进行全身关节角度追踪
        activate_snooker_laplacian: bool,  # 是否添加 snooker 虚拟点到 Laplacian 网格
        # 以下参数有默认值
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        visualize: bool = False,
        debug: bool = False,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
        snooker_frame_range: list[int] | None = None, 
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: the index in robot's configuration where the optimization variables start. -7: starts from the
            floating base, -3: starts from the translation of the floating base, 0: starts from the actuated DOF,
            12: starts from waist, 15: starts from left shoulder
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision
            when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            foot_sticking_tolerance: tolerance for foot sticking constraints in x, y.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
            snooker_frame_range: [start_frame, end_frame] where snooker constraints are active.
        """

        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        self.object_model_path = object_urdf_path
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS, task_constants.FOOT_STICKING_LINKS))
        self.penetration_tolerance = penetration_tolerance
        self.step_size = step_size
        self.visualize = visualize
        self.debug = debug
        self.demo_joints = task_constants.DEMO_JOINTS
        self.task_constants = task_constants

        # 基础 Laplacian 映射来自配置（不包含 snooker 虚拟点）
        self.base_laplacian_match_links = task_constants.JOINTS_MAPPING
        self.base_link_keys = list(self.base_laplacian_match_links.keys())
        self.base_key_to_idx = {k: i for i, k in enumerate(self.base_link_keys)}

        # Snooker 虚拟点仅在本文件内定义，避免污染 data_type
        self.snooker_virtual_links = {
            "LeftHandBridge": "left_wrist_yaw_link",
            "RightHandGrip": "right_wrist_yaw_link",
            "CueTip": "right_wrist_yaw_link",
        }
        self.virtual_site_offsets = {
            "LeftHandBridge": np.array([0.0415, 0.003, 0.0], dtype=float),
            "RightHandGrip": np.array([0.0415, -0.003, 0.0], dtype=float),
            "CueTip": np.array([0.1215, 0.017, 1.075], dtype=float),
        }
        self.snooker_cue_grip_offset = np.array([0.1215, 0.017, 0.0], dtype=float)
        self.snooker_cue_length = 1.075

        # 保持旧变量名用于兼容其他逻辑（仅含基础映射）
        self.laplacian_match_links = self.base_laplacian_match_links

        self.snooker_frame_range = snooker_frame_range
        self.snooker_ramp_frames = 45  # 过渡帧数增加到 45 帧（约 1.5 秒），使切换更缓慢平滑
        self.activate_snooker_tracking = activate_snooker_tracking
        self.activate_snooker_laplacian = activate_snooker_laplacian
        self.activate_realtime_rotation_tracking = activate_realtime_rotation_tracking
        self.activate_general_nominal_tracking = activate_general_nominal_tracking

        # --- 初始化日志功能 ---
        # 使用相对于脚本所在目录的路径，确保存入 holosoma_retargeting/logs
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"retarget_debug_{timestamp}.log"
        with open(self.log_path, "w") as f:
            f.write(f"=== Retargeting Debug Log - Started at {datetime.now()} ===\n")
            f.write(f"Parameters:\n")
            f.write(f"- activate_snooker_tracking: {self.activate_snooker_tracking}\n")
            f.write(f"- activate_snooker_laplacian: {self.activate_snooker_laplacian}\n")
            f.write(f"- activate_realtime_rotation_tracking: {self.activate_realtime_rotation_tracking}\n")
            f.write(f"- activate_general_nominal_tracking: {self.activate_general_nominal_tracking}\n\n")

        self.smplh_mapped_joint_indices = [self.demo_joints.index(name) for name in self.base_link_keys]
        self.left_hand_idx = self.demo_joints.index("LeftHand") if "LeftHand" in self.demo_joints else None
        self.right_hand_idx = self.demo_joints.index("RightHand") if "RightHand" in self.demo_joints else None
        self.has_snooker_hands = (self.left_hand_idx is not None) and (self.right_hand_idx is not None)

        # Setup weights and parameters
        self.laplacian_weights = 10
        self.smooth_weight = 0.2
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = foot_sticking_tolerance

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization()

        # Load Mujoco model
        if self.object_name == "ground":
            robot_xml_path = self.robot_model_path.replace(".urdf", ".xml")
        #! scene
        elif hasattr(self.task_constants, 'SCENE_XML_FILE') and self.task_constants.SCENE_XML_FILE:
            # Use SCENE_XML_FILE if set (for climbing tasks with custom objects like multi_boxes, snooker_table, etc.)
            robot_xml_path = self.task_constants.SCENE_XML_FILE
        else:
            robot_xml_path = self.robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)

        if self.robot_data.qpos.shape[0] > 7 + self.task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        self.nq = self.robot_model.nq

        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + self.q_a_init_idx, 7 + self.task_constants.ROBOT_DOF)

        self.nq_a = len(self.q_a_indices)

        # Create complete limits with floating base (-inf, inf) and actuated joint limits
        n_floating_base = 7
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]  # Filter out None names

        large_number = 1e6
        complete_lower_limits = np.concatenate(
            [-large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 0]]
        )
        complete_upper_limits = np.concatenate(
            [large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 1]]
        )

        self.q_a_lb = complete_lower_limits[self.q_a_indices]
        self.q_a_ub = complete_upper_limits[self.q_a_indices]

        self.q_a_lb[np.array(list(self.task_constants.MANUAL_LB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_LB.values()
        )
        self.q_a_ub[np.array(list(self.task_constants.MANUAL_UB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_UB.values()
        )

        #! cube: Override wrist limits for snooker to prevent shaking
        # Note: task_constants is a SimpleNamespace populated with UPPERCASE properties from RobotConfig.
        # robot_type is lowercase, so it's not copied. We use ROBOT_NAME instead.
        if self.snooker_frame_range is not None and "g1" in getattr(self.task_constants, "ROBOT_NAME", "").lower():
            # 分开处理左右手腕：左手(26,27,28)，右手(33,34,35)
            left_wrist_indices = [26, 27, 28]
            right_wrist_indices = [33, 34, 35]
            
            # 右手（握杆手）：完全放开限制，保证推杆和瞄准角度的灵活性
            print(f"Releasing right wrist limits for Snooker task: {right_wrist_indices}")
            for idx in right_wrist_indices:
                if idx < len(self.q_a_lb):
                    self.q_a_lb[idx] = complete_lower_limits[idx]
                    self.q_a_ub[idx] = complete_upper_limits[idx]
            
            # 左手（架杆手）：同样恢复物理范围，改用 Nominal Tracking 进行姿态引导
            print(f"Releasing left wrist limits for Snooker task (using soft tracking): {left_wrist_indices}")
            for idx in left_wrist_indices:
                if idx < len(self.q_a_lb):
                    self.q_a_lb[idx] = complete_lower_limits[idx]
                    self.q_a_ub[idx] = complete_upper_limits[idx]

        # Prevent too much waist twist
        self.Q_diag = np.zeros(self.nq_a) * 1e-3
        self.Q_diag[np.array(list(self.task_constants.MANUAL_COST.keys())).astype(int)] = list(
            self.task_constants.MANUAL_COST.values()
        )

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = task_constants.NOMINAL_TRACKING_INDICES

    #! cube
    def _calc_snooker_alpha(self, frame_idx: int) -> float:
        """Compute snooker activation alpha for a given frame index using cosine smoothing."""
        if self.snooker_frame_range is None:
            return 0.0
        start_f, end_f = self.snooker_frame_range
        if (frame_idx < start_f) or (frame_idx > end_f):
            return 0.0
        ramp = max(int(self.snooker_ramp_frames), 1)
        # 计算基础线性进度 t (0.0 to 1.0)
        if frame_idx < start_f + ramp:
            t = (frame_idx - start_f) / ramp
        elif frame_idx > end_f - ramp:
            t = (end_f - frame_idx) / ramp
        else:
            return 1.0
        # 应用余弦平滑函数 (Smoothstep-like): 0.5 * (1 - cos(pi * t))
        # 这种曲线在起点和终点的导数为 0，能实现无感的力平滑过渡。
        return 0.5 * (1.0 - np.cos(np.pi * t))

    #! cube：整个函数都是新增的。
    def _get_active_laplacian_links(self, frame_idx: int) -> tuple[dict[str, str], float]:
        """Return active Laplacian links and snooker alpha for the given frame."""
        snooker_alpha = self._calc_snooker_alpha(frame_idx)  # 只有传入 frame_range 才会>0

        active_links = dict(self.base_laplacian_match_links)

        # 只有开启 snooker Laplacian 且具备左右手数据时，才追加虚拟点
        if self.activate_snooker_laplacian and self.has_snooker_hands:
            if self.snooker_frame_range is None:
                active_links.update(self.snooker_virtual_links)
            elif snooker_alpha > 0:
                active_links.update(self.snooker_virtual_links)

        return active_links, snooker_alpha

    def _compute_snooker_virtual_positions(
        self,
        frame_idx: int,
        human_pos_full: np.ndarray,
        human_quat_full: np.ndarray | None,
        has_rot_data: bool,
    ) -> dict[str, np.ndarray]:
        """Compute snooker virtual points in human space for the given frame."""
        if not self.has_snooker_hands:
            return {}

        lh_pos = human_pos_full[frame_idx, self.left_hand_idx]
        rh_pos = human_pos_full[frame_idx, self.right_hand_idx]

        r_lh = None
        r_rh = None
        if has_rot_data and human_quat_full is not None:
            lh_quat_wxyz = human_quat_full[frame_idx, self.left_hand_idx]
            rh_quat_wxyz = human_quat_full[frame_idx, self.right_hand_idx]
            r_lh = Rotation.from_quat([lh_quat_wxyz[1], lh_quat_wxyz[2], lh_quat_wxyz[3], lh_quat_wxyz[0]])
            r_rh = Rotation.from_quat([rh_quat_wxyz[1], rh_quat_wxyz[2], rh_quat_wxyz[3], rh_quat_wxyz[0]])

        if r_lh is not None and r_rh is not None:
            left_bridge = lh_pos + r_lh.apply(self.virtual_site_offsets["LeftHandBridge"])
            right_grip = rh_pos + r_rh.apply(self.virtual_site_offsets["RightHandGrip"])
            cue_grip_on_stick = rh_pos + r_rh.apply(self.snooker_cue_grip_offset)
            bridge_pos = lh_pos + r_lh.apply(self.virtual_site_offsets["LeftHandBridge"])
        else:
            left_bridge = lh_pos + self.virtual_site_offsets["LeftHandBridge"]
            right_grip = rh_pos + self.virtual_site_offsets["RightHandGrip"]
            cue_grip_on_stick = rh_pos + self.snooker_cue_grip_offset
            bridge_pos = lh_pos + self.virtual_site_offsets["LeftHandBridge"]

        direction = bridge_pos - cue_grip_on_stick
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            if r_rh is not None:
                direction_norm = r_rh.apply(np.array([0.0, 0.0, 1.0]))
            else:
                direction_norm = np.array([0.0, 0.0, 1.0])
        else:
            direction_norm = direction / norm

        cue_tip = cue_grip_on_stick + direction_norm * self.snooker_cue_length

        return {
            "LeftHandBridge": left_bridge,
            "RightHandGrip": right_grip,
            "CueTip": cue_tip,
        }

    def _setup_visualization(self):
        """Setup Viser visualization components."""
        self.server = viser.ViserServer()

        # 1) Ensure a world frame exists (absolute path!)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        # Create parent frames for robot and object
        self.robot_base = self.server.scene.add_frame("/world/robot", show_axes=False)

        print("robot_model_path: ", self.robot_model_path)

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(
            self.robot_model_path,
            load_meshes=True,
            build_scene_graph=True,
        )

        print("Viser using robot URDF: ", self.robot_model_path)

        # Create ViserUrdf instance for robot, attaching it to the robot_base frame
        self.viser_robot = ViserUrdf(
            self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",  # This links to the robot_base frame we created
        )

        # Similarly for object
        if self.object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)

            self.object_urdf = yourdfpy.URDF.load(
                self.object_model_path,
                load_meshes=True,
                build_scene_graph=True,
            )

            # Create ViserUrdf instance for object, attaching it to the object_base frame
            self.viser_object = ViserUrdf(
                self.server,
                urdf_or_path=self.object_urdf,
                root_node_name="/world/object",  # This links to the object_base frame we created
            )
            print("Viser using object URDF: ", self.object_model_path)

        else:
            self.viser_object = None

        # Check the number of actuated joints and their names
        robot_joint_limits = self.viser_robot.get_actuated_joint_limits()
        print("\nRobot joints:")
        print("Number of actuated joints:", len(robot_joint_limits))
        print("Joint names:", list(robot_joint_limits.keys()))

        # Initialize robot with this configuration
        robot_initial_config = np.zeros(len(robot_joint_limits))
        self.viser_robot.update_cfg(robot_initial_config)

        # Add grid
        self.server.scene.add_grid(
            "/world/grid",
            width=8,
            height=8,
            position=(0.0, 0.0, 0.0),
        )

    def draw_mesh_from_geom(self, model, data, geom_id, geom_name, name="/mesh", color=(50, 150, 255), opacity=0.5):
        """
        Draw a single MuJoCo mesh geom (already baked to world coords) in viser.
        color is [0, 255] RGB ints; opacity is [0,1].
        """
        if not hasattr(self, "server"):
            return
        V, F = _world_mesh_from_geom(model, data, geom_id, geom_name)
        self.server.scene.add_mesh_simple(
            name,
            vertices=V.astype(np.float32),
            faces=F.astype(np.int32),
            position=(0.0, 0.0, 0.0),  # already world-frame
            color=tuple(int(c) for c in color),
            opacity=float(opacity),
        )

    def draw_mesh_pair_with_contact(
        self,
        model,
        data,
        geom_id1,
        geom_id2,
        geom1_name,
        geom2_name,
        fromto=None,
        group_name="pair",
        color1=(50, 150, 255),
        color2=(255, 120, 60),
        opacity=0.45,
        show_segment=True,
    ):
        """
        Draw two meshes and (optionally) a contact/query segment.
        Uses the existing self.draw_keypoints(...) to visualize points.
        """
        # Note: sometime geom does not have mesh, mesh_id will be -1
        if int(model.geom_dataid[geom_id1]) == -1 or int(model.geom_dataid[geom_id2]) == -1:
            return

        base = f"/{group_name}"
        # meshes
        self.draw_mesh_from_geom(model, data, geom_id1, geom1_name, name=f"{base}/mesh1", color=color1, opacity=opacity)
        self.draw_mesh_from_geom(model, data, geom_id2, geom2_name, name=f"{base}/mesh2", color=color2, opacity=opacity)

        # contact points (q: green, c: red) via your draw_keypoints
        if fromto is not None:
            q = np.asarray(fromto[:3], dtype=float)
            c = np.asarray(fromto[3:], dtype=float)

            # your existing helper (rgba expects floats 0..1)
            self.draw_keypoints(q, name=f"{group_name}_q", rgba=(0.0, 1.0, 0.0, 1.0))
            self.draw_keypoints(c, name=f"{group_name}_c", rgba=(1.0, 0.0, 0.0, 1.0))

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        foot_sticking_sequences,
        q_a_init=None,
        q_nominal_list=None,
        original=True,
        dest_res_path=None,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3 or 7) array.
                                              If 7, it contains (pos, quat).
            object_poses (np.ndarray): (num_frames, 7) array of demo object poses (quat, trans).
            object_poses_augmented (np.ndarray): (num_frames, 7) array of augmented object poses (quat, trans).
            object_points_local_demo (np.ndarray): Demo object points in local frame (rest pose).
            object_points_local (np.ndarray): Current object points in local frame (rest pose).
            foot_sticking_sequences (list): List of foot sticking sequences for each frame.
            q_a_init (np.ndarray, optional): Initial robot configuration.
            q_nominal_list (np.ndarray, optional): Nominal robot configuration.

        Returns:
            tuple: (retargeted_motions, obj_pts_demo_list, obj_pts_list, tetrahedra)
        """
        num_frames = human_joint_motions.shape[0]
        
        #! cube: Handle 7D data (pos + quat) for Nominal Tracking
        # If human_joint_motions is (T, J, 7), extract positions and rotations
        if human_joint_motions.shape[-1] == 7:
            human_pos_full = human_joint_motions[..., :3]
            human_quat_full = human_joint_motions[..., 3:]
            has_rot_data = True
        else:
            human_pos_full = human_joint_motions
            human_quat_full = None
            has_rot_data = False

        if q_nominal_list is not None:
            q_locked_list = q_nominal_list
        else:
            q_locked_list = np.zeros((num_frames, self.nq))
            q_locked_list[0, self.q_a_indices] = q_a_init

        q_locked_list[:, -7:] = object_poses_augmented
        q = np.copy(q_locked_list[0])
        retargeted_motions = [q]

        tetrahedra = []
        obj_pts_demo_list = []  # scaled object pts
        obj_pts_list = []  # original size object pts

        print(f"\nStarting motion retargeting for {num_frames} frames...")

        with tqdm(range(num_frames)) as pbar: #初始化一个进度条（tqdm），总长度为总帧数 num_frames，用于可视化 retarget 进度
            for i in pbar: #i是当前帧的索引
                # Get object poses and transform points
                object_quat_demo = object_poses[i, 3:]
                object_trans_demo = object_poses[i, :3]

                #! cube:  Get human joint positions and create interaction mesh in object frame
                # 根据基础映射提取当前帧的人体关节点（不包含 snooker 虚拟点）
                base_human_joints = human_pos_full[i, self.smplh_mapped_joint_indices]

                # 计算本帧活跃的网格顶点（按顺序返回）
                active_links, _snooker_alpha = self._get_active_laplacian_links(i)
                active_link_keys = list(active_links.keys())
                active_link_names = list(active_links.values())

                # 如果启用 snooker 虚拟点，则在本地按顺序拼接
                snooker_virtual_positions = {}
                if self.activate_snooker_laplacian and self.has_snooker_hands:
                    snooker_virtual_positions = self._compute_snooker_virtual_positions(
                        i, human_pos_full, human_quat_full, has_rot_data
                    )

                human_mapped_joints = []
                for key in active_link_keys:
                    if key in self.base_key_to_idx:
                        human_mapped_joints.append(base_human_joints[self.base_key_to_idx[key]])
                    else:
                        human_mapped_joints.append(snooker_virtual_positions.get(key, np.zeros(3)))

                human_mapped_joints = np.asarray(human_mapped_joints, dtype=float)

                if self.object_name == "ground": #如果操作物体是“地面”，则认为物体的局部坐标系就是世界坐标系，无需坐标变换。
                    human_mapped_joints_in_object = human_mapped_joints
                else: #如果操作的是特定物体（如台球桌），则将人体关节点的坐标从世界坐标系转换到该物体的局部坐标系下。这是交互网格的核心：所有的几何关系都是相对于物体定义的。
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo, object_trans_demo, human_mapped_joints
                    )

                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack([human_mapped_joints_in_object, object_points_local_demo]) #将转换后的人体关节点和物体表面采样点（object_points_local_demo）合并成一个大的点云。
                ) #create_interaction_mesh 对这个点云进行 Delaunay 三角剖分，生成四面体网格
                tetrahedra.append(source_tetrahedra)


                if self.debug:
                    # Only for visualization
                    object_quat = object_poses_augmented[i, 3:]
                    object_trans = object_poses_augmented[i, :3]
                    obj_pts_demo = transform_points_local_to_world(
                        object_quat_demo, object_trans_demo, object_points_local_demo
                    )
                    obj_pts = transform_points_local_to_world(object_quat, object_trans, object_points_local)

                    obj_pts_demo_list.append(obj_pts_demo)
                    obj_pts_list.append(obj_pts)
                    human_kpts_handle_list = self.draw_keypoints(human_mapped_joints, name="human_kpts")  # 15 X 3
                    obj_kpts_demo_handle_list = self.draw_keypoints(
                        obj_pts_demo, name="object_demo_kpts", rgba=(1, 0, 0, 1)
                    )  # 100 X 3
                    obj_kpts_handle_list = self.draw_keypoints(
                        obj_pts, name="object_kpts", rgba=(0, 1, 1, 1)
                    )  # 100 X 3

                # Create adjacency list and calculate target Laplacian coordinates
                adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)

                # Run optimization
                if original:
                    w_nominal_tracking = self.w_nominal_tracking_init
                else:
                    w_nominal_tracking = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

                #! cube: 构造实时姿态引导目标 (Real-time Rotation Tracking)
                # 如果没有外部提供的 q_nominal_list，且开启了实时姿态追踪，则从 7D 数据中提取
                curr_q_a_nominal = None
                is_full_nominal = False # 标记是否是完整的参考序列

                
                
                if q_nominal_list is not None: #q_nominal_list只会从*_original.npz文件中获取，并不会从输入的npy文件中获取； 而且q_nominal_list存储的是每个关节的旋转角度。
                    curr_q_a_nominal = q_nominal_list[i, self.q_a_indices]
                    is_full_nominal = True
                #! wrist: [DEPRECATED] 原方案 - 构造局部关节角度目标（已被全局旋转 tracking 替代）
                # elif has_rot_data and self.activate_snooker_tracking:
                #     #! wrist: 从 7D 数据中实时提取人类左手旋转，映射到机器人左手腕
                #     # 注意：这里只设置左手腕的目标值，其他关节保持 None（不会被追踪）
                #     lw_yaw_idx_global = 28  # G1 左手腕 Yaw 关节的全局索引
                #     if lw_yaw_idx_global in self.q_a_indices and "LeftHand" in self.demo_joints:
                #         # 构造一个只包含左手腕目标的数组
                #         curr_q_a_nominal = np.zeros(self.nq_a)
                #         
                #         # BUG FIX: 从 demo_joints 中找到 LeftHand 的索引（对应 human_quat_full 的第二维）
                #         lh_idx_in_demo = self.demo_joints.index("LeftHand")
                #         lh_quat_wxyz = human_quat_full[i, lh_idx_in_demo]  # (w, x, y, z)
                #         
                #         # 将四元数转为欧拉角（scipy 用 xyzw 顺序）
                #         r_lh = Rotation.from_quat([lh_quat_wxyz[1], lh_quat_wxyz[2], lh_quat_wxyz[3], lh_quat_wxyz[0]])
                #         lh_euler = r_lh.as_euler('xyz')  # [roll, pitch, yaw]
                #         
                #         # 将人类 LeftHand 的 yaw 分量映射到机器人左手腕 Yaw 关节
                #         lw_local_idx = np.where(self.q_a_indices == lw_yaw_idx_global)[0][0]
                #         curr_q_a_nominal[lw_local_idx] = lh_euler[2]  # yaw 角度
                #         
                #         if i == 0:
                #             debug_msg = (
                #                 f"\n=== [Snooker Tracking Data Extraction - Frame {i}] ===\n"
                #                 f"  demo_joints: {self.demo_joints}\n"
                #                 f"  LeftHand idx in demo_joints: {lh_idx_in_demo}\n"
                #                 f"  human_quat_full shape: {human_quat_full.shape}\n"
                #                 f"  LeftHand quat (wxyz): {lh_quat_wxyz}\n"
                #                 f"  LeftHand euler (xyz, deg): roll={np.rad2deg(lh_euler[0]):.2f}, pitch={np.rad2deg(lh_euler[1]):.2f}, yaw={np.rad2deg(lh_euler[2]):.2f}\n"
                #                 f"  lw_local_idx in q_a: {lw_local_idx}\n"
                #                 f"  curr_q_a_nominal[lw_local_idx]: {curr_q_a_nominal[lw_local_idx]:.6f} rad = {np.rad2deg(curr_q_a_nominal[lw_local_idx]):.2f} deg\n"
                #             )
                #             print(debug_msg)
                #             with open(self.log_path, "a") as f:
                #                 f.write(debug_msg)

                #安全检查 - 确保不会误用错误的 nominal tracking
                # 当 q_nominal_list 为 None 时，curr_q_a_nominal 应该保持为 None
                # 旧方案（局部关节角度 tracking）已被注释，全局旋转 tracking 使用 target_lh_quat 而非 curr_q_a_nominal
                if i == 0:
                    if q_nominal_list is None:
                        print("[Info] q_nominal_list is None - 使用全局旋转 tracking（不使用 curr_q_a_nominal）")
                    if curr_q_a_nominal is not None and not is_full_nominal:
                        # 这种情况不应该发生（旧方案已被注释），如果发生则警告
                        print("[警示] curr_q_a_nominal 被设置但不是完整参考序列，可能存在逻辑错误！")
                        with open(self.log_path, "a") as f:
                            f.write(f"[WARNING Frame {i}] curr_q_a_nominal 被设置但 is_full_nominal=False\n")

                #! wrist: 提取人类 LeftHand 的全局四元数用于全局旋转 tracking
                target_lh_quat = None
                if has_rot_data and self.activate_snooker_tracking and "LeftHand" in self.demo_joints:
                    lh_idx_in_demo = self.demo_joints.index("LeftHand")
                    target_lh_quat = human_quat_full[i, lh_idx_in_demo]  # (w, x, y, z) 全局四元数

                q, cost = self.iterate(
                    q_locked=q_locked_list[i],
                    q_n=q,
                    q_t_last=retargeted_motions[-1],
                    target_laplacian=target_laplacian,
                    adj_list=adj_list,
                    obj_pts_local=object_points_local,
                    foot_sticking=foot_sticking_sequences[i],
                    w_nominal_tracking=w_nominal_tracking,
                    q_a_nominal=curr_q_a_nominal, #! cube: 用于Nominal Tracking
                    init_t=i == 0,
                    n_iter=50 if i == 0 else 10,
                    frame_idx=i,
                    is_full_nominal=is_full_nominal, #! cube: 传递标记位
                    target_lh_quat=target_lh_quat, #! wrist: 传递目标全局四元数
                )
                if self.debug:
                    robot_link_positions = self._get_robot_link_positions(
                        # q, self.laplacian_match_links.values() #! cube
                        q, active_link_names #! cube
                    )  # 15 X 3
                    robot_kpts_handle_list = self.draw_keypoints(
                        robot_link_positions, name="robot_kpts", rgba=(0, 1, 0, 1)
                    )

                retargeted_motions.append(q)
                if self.visualize and self.debug:
                    self.draw_q(q)

                pbar.set_postfix(cost=cost)

        # Remove previous debug visualization
        if self.debug:
            for handle in human_kpts_handle_list:
                handle.remove()
            human_kpts_handle_list.clear()

            for handle in obj_kpts_demo_handle_list:
                handle.remove()
            obj_kpts_demo_handle_list.clear()

            for handle in obj_kpts_handle_list:
                handle.remove()
            obj_kpts_handle_list.clear()

            for handle in robot_kpts_handle_list:
                handle.remove()
            robot_kpts_handle_list.clear()

        # Save results
        np.savez(
            dest_res_path,
            qpos=np.array(retargeted_motions)[1:],
            human_joints=human_joint_motions,
            fps=30,
            cost=cost,
        )
        print("Saving results to path:", dest_res_path)

        if self.visualize:
            robot_dof = len(self.viser_robot.get_actuated_joint_limits())

            create_motion_control_sliders(
                server=self.server,
                viser_robot=self.viser_robot,
                robot_base_frame=self.robot_base,
                motion_sequence=np.asarray(retargeted_motions)[1:],
                robot_dof=robot_dof,
                viser_object=self.viser_object,
                object_base_frame=getattr(self, "object_base", None) if self.viser_object else None,
                contains_object_in_qpos=bool(self.viser_object) and bool(self.has_dynamic_object),
                initial_fps=30,
                initial_interp_mult=2,
                loop=False,
            )

            # 4) optional: visibility toggle
            with self.server.gui.add_folder("Visibility"):
                show_meshes_cb = self.server.gui.add_checkbox("Show meshes", self.viser_robot.show_visual)

                @show_meshes_cb.on_update
                def _(_):
                    self.viser_robot.show_visual = show_meshes_cb.value
                    if self.viser_object is not None:
                        self.viser_object.show_visual = show_meshes_cb.value

        return (
            np.array(retargeted_motions)[1:],
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
        )

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
        frame_idx: int = 0,
        is_full_nominal: bool = False,
        target_lh_quat: np.ndarray | None = None,  #! wrist: 新增 - 目标全局四元数 (wxyz)
    ):
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            foot_sticking: a sequence of booleans indicating whether the foot [left, right] is sticking to the ground.
            smpl_joints: the (possibly scaled) SMPL joint positions to match for IK.
            q_ref: the reference robot configuration.
            smpl_joints_original: the original SMPL joint positions (used for contact matching).
            obj_original: the original object pose (used for contact matching).
            init_t: the current time step is the first time step.
        """
        assert len(q_a_n_last) == self.nq_a

        # Lock the object pose and set the current robot slice to last accepted solution
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        #! cube: Snooker activity logic (Frame Gating + Smooth Transition)
        active_links, snooker_alpha = self._get_active_laplacian_links(frame_idx) #数学求解阶段（每帧可能会跑多次，在优化器迭代中）。

        # 调试打印：确认网格顶点和 Tracking 状态
        # 强制在第 0 帧打印，后续每 100 帧在 debug 模式下打印
        if (frame_idx == 0) or (self.debug and frame_idx % 100 == 0):
            print(f"\n--- [DEBUG Frame {frame_idx}] ---")
            robot_keys = list(active_links.keys())
            print(f"Laplacian Nodes (Total {len(robot_keys)}): {robot_keys}")
            
            # 检查 Nominal Tracking 状态
            snooker_track_on = self.activate_snooker_tracking and snooker_alpha > 0 and q_a_nominal is not None
            general_track_on = self.activate_general_nominal_tracking and is_full_nominal and (w_nominal_tracking > 0) and (q_a_nominal is not None)
            
            print(f"Nominal Tracking Status:")
            print(f"  - Snooker Tracking (Left Wrist): {'ON' if snooker_track_on else 'OFF'} (alpha: {snooker_alpha:.2f})")
            print(f"  - General Nominal Tracking: {'ON' if general_track_on else 'OFF'}")
            if general_track_on:
                print(f"    - Tracked Indices: {self.track_nominal_indices}")

        # Compute Laplacian pieces
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            # q, links=self.laplacian_match_links, obj_frame=(self.object_name != "ground")  #! cube
            q, links=active_links, obj_frame=(self.object_name != "ground") #! cube
        )
        # robot_link_keys = list(self.laplacian_match_links.keys()) #! cube
        robot_link_keys = list(active_links.keys()) #! cube
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians for robot points
        J_V = np.zeros((3 * V, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
        vertices = np.vstack([robot_pts_local, obj_pts_local])  # (V x 3)

        #! cube:  Snooker-specific: add manual edges and adjust weights
        if self.activate_snooker_laplacian and snooker_alpha > 0 and ("RightHandGrip" in robot_link_keys) and ("LeftHandBridge" in robot_link_keys) and ("CueTip" in robot_link_keys):
            # 获取这三个关键点在网格顶点列表中的索引
            rh_grip_idx = robot_link_keys.index("RightHandGrip")
            lh_bridge_idx = robot_link_keys.index("LeftHandBridge")
            cue_tip_idx = robot_link_keys.index("CueTip")
            
            # 在邻接表（adj_list）中增加边：右手握杆点 <-> 左手架杆点
            if lh_bridge_idx not in adj_list[rh_grip_idx]:
                adj_list[rh_grip_idx].append(lh_bridge_idx) #adj_list[i] 存储了所有与第 i 个点直接相连的点的索引。
            if rh_grip_idx not in adj_list[lh_bridge_idx]:
                adj_list[lh_bridge_idx].append(rh_grip_idx)
                
            # 在邻接表（adj_list）中增加边：左手架杆点 <-> 球杆尖端  
            if cue_tip_idx not in adj_list[lh_bridge_idx]:
                adj_list[lh_bridge_idx].append(cue_tip_idx)
            if lh_bridge_idx not in adj_list[cue_tip_idx]:
                adj_list[cue_tip_idx].append(lh_bridge_idx)

        # --- 增加严谨的调试逻辑：写入日志文件 ---
        if (frame_idx == 0) or (self.debug and frame_idx % 100 == 0):
            with open(self.log_path, "a") as f:
                f.write(f"\n--- [LAPLACIAN MESH DETAIL - Frame {frame_idx}] ---\n")
                # 1. 写入顶点信息
                f.write(f"{'Index':<6} | {'Link/Point Name':<20} | {'Position (x, y, z)':<25}\n")
                f.write("-" * 60 + "\n")
                # 机器人点
                for idx, key in enumerate(robot_link_keys):
                    pos = vertices[idx]
                    f.write(f"{idx:<6} | {key:<20} | [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]\n")
                # 物体点
                for idx_o in range(V_o):
                    idx = V_r + idx_o
                    pos = vertices[idx]
                    f.write(f"{idx:<6} | {'ObjectPoint_' + str(idx_o):<20} | [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]\n")
                
                # 2. 写入邻接关系
                f.write("\n--- Adjacency List (Topology) ---\n")
                idx_to_name = list(robot_link_keys) + [f"Obj_{j}" for j in range(V_o)]
                for idx, neighbors in enumerate(adj_list):
                    name = idx_to_name[idx]
                    neighbor_names = [idx_to_name[n_idx] for n_idx in neighbors]
                    f.write(f"{name:<20} ({idx:<2}) -> Connected to: {neighbor_names}\n")
                f.write("-" * 60 + "\n")
            
            # 终端保持简洁提醒
            if frame_idx == 0:
                print(f"[*] Debug details will be saved to: {self.log_path}")
            print(f"[Frame {frame_idx}] Laplacian mesh detail saved to log.")

        L = calculate_laplacian_matrix(vertices, adj_list)  # (V x V)
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V

        lap0 = L @ vertices
        lap0_vec = lap0.reshape(-1)  # (3V,)
        target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

        # 计算 Laplacian 权重
        w_v = (self.laplacian_weights * np.ones(V)).astype(float)
        
        #! cube  : 
        if self.activate_snooker_laplacian and snooker_alpha > 0:
            # 权重随 alpha 平滑增强
            snooker_weight = 0 + (10 * snooker_alpha) 
            w_v[rh_grip_idx] = snooker_weight
            w_v[lh_bridge_idx] = snooker_weight
            w_v[cue_tip_idx] = snooker_weight

        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        # Constraints list
        constraints = []

        # Linear equality
        constraints += [cp.Constant(J_L[:, self.q_a_indices]) @ dqa - lap_var == -lap0_vec]

        # Foot sticking
        if (self.q_a_init_idx < 12) and self.activate_foot_sticking:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)
            _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(q_t_last, links=self.foot_links, obj_frame=False)
            # Identify 'left' and 'right' flags from provided keys
            left_key = right_key = None
            for key in foot_sticking:
                if key.lower().startswith("l"):
                    left_key = key
                elif key.lower().startswith("r"):
                    right_key = key
            if left_key is None or right_key is None:
                raise ValueError("foot_sticking must include one left* and one right* key")

            for key, J_WF in J_WF_dict.items():
                apply_left = ("left" in key) and foot_sticking[left_key]
                apply_right = ("right" in key) and foot_sticking[right_key]
                if apply_left or apply_right:
                    p_lb = p_WF_t_last_dict[key] - p_WF_dict[key] - self.foot_sticking_tolerance
                    p_ub = p_lb + 2 * self.foot_sticking_tolerance  # symmetric window

                    Jxy = J_WF[:2, self.q_a_indices]  # (2 x nq_act)
                    constraints += [
                        Jxy @ dqa >= p_lb[:2],
                        Jxy @ dqa <= p_ub[:2],
                    ]

        # Non-penetration constraints
        Js, phis = self._update_jacobians_and_phis_from_q(q)
        for key, phi in phis.items():
            Ja_n_full = Js[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            rhs = -phi - self.penetration_tolerance
            constraints += [Ja_n @ dqa >= rhs]

        # Joint limits constraints (actuated)
        if self.activate_joint_limits:
            constraints += [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
            ]

        # Step size constraints (Lorentz cone)
        constraints += [cp.SOC(self.step_size, dqa)]

        # objective
        obj_terms = []
        obj_names = [] # 新增：用于记录各分量名称

        #! wrist: [DEPRECATED] 原方案 - 追踪局部关节角度（已被全局旋转 tracking 替代）
        # 问题：直接把人类全局 yaw 赋给机器人局部 yaw 是错误的映射
        # 现在由下方的 global_rotation_tracking 替代
        # snooker_tracking_added = False
        # if self.activate_snooker_tracking and snooker_alpha > 0 and q_a_nominal is not None:
        #     lw_yaw_idx_global = 28  # G1 左手腕 Yaw 的全局关节索引
        #     if lw_yaw_idx_global in self.q_a_indices:
        #         # 将全局索引转换为优化变量数组中的局部索引
        #         lw_local_idx = np.where(self.q_a_indices == lw_yaw_idx_global)[0][0]
        #         lw_tracking_weight = 5.0 * snooker_alpha
        #         
        #         target_val = q_a_nominal[lw_local_idx]
        #         curr_val = q_a_n_last[lw_local_idx]
        #         delta = target_val - curr_val
        #         
        #         # 目标：让 dqa 驱动关节角逼近 q_a_nominal 中的目标值
        #         error = dqa[lw_local_idx] - delta
        #         term = lw_tracking_weight * cp.sum_squares(error)
        #         obj_terms.append(term)
        #         obj_names.append("snooker_tracking")
        #         snooker_tracking_added = True
        #         
        #         # 详细调试信息写入日志
        #         if (frame_idx == 0) or (self.debug and frame_idx % 100 == 0):
        #             debug_msg = (
        #                 f"\n=== [Snooker Tracking Debug - Frame {frame_idx}] ===\n"
        #                 f"  snooker_alpha: {snooker_alpha:.4f}\n"
        #                 f"  lw_local_idx: {lw_local_idx}\n"
        #                 f"  lw_tracking_weight: {lw_tracking_weight:.4f}\n"
        #                 f"  q_a_nominal[lw_local_idx] (target): {target_val:.6f} rad = {np.rad2deg(target_val):.2f} deg\n"
        #                 f"  q_a_n_last[lw_local_idx] (current): {curr_val:.6f} rad = {np.rad2deg(curr_val):.2f} deg\n"
        #                 f"  delta (target - current): {delta:.6f} rad = {np.rad2deg(delta):.2f} deg\n"
        #                 f"  expected tracking cost: {lw_tracking_weight * delta**2:.6f}\n"
        #             )
        #             print(debug_msg)
        #             with open(self.log_path, "a") as f:
        #                 f.write(debug_msg)
        # 
        # # 如果 tracking 没有被添加，记录原因
        # if (frame_idx == 0) and not snooker_tracking_added:
        #     reason = []
        #     if not self.activate_snooker_tracking:
        #         reason.append("activate_snooker_tracking=False")
        #     if snooker_alpha <= 0:
        #         reason.append(f"snooker_alpha={snooker_alpha}<=0")
        #     if q_a_nominal is None:
        #         reason.append("q_a_nominal is None")
        #     debug_msg = f"\n[Frame {frame_idx}] Snooker tracking NOT added. Reasons: {', '.join(reason)}\n"
        #     print(debug_msg)
        #     with open(self.log_path, "a") as f:
        #         f.write(debug_msg)

        #! wrist:
        # 使用旋转雅可比匹配机器人 left_wrist_yaw_link 的全局旋转到人类 LeftHand 的全局旋转
        global_rotation_tracking_added = False
        if self.activate_snooker_tracking and snooker_alpha > 0 and target_lh_quat is not None:
            try:
                # 计算旋转雅可比和误差
                J_rot, rot_error, current_quat = self._calc_rotation_jacobian_and_error(
                    q, "left_wrist_yaw_link", target_lh_quat
                )
                
                # 线性化：rot_error_new ≈ rot_error - J_rot @ dqa
                # Cost: ||rot_error - J_rot @ dqa||^2
                # 最小化 dqa 使旋转误差趋近于 0
                rotation_tracking_weight = 10.0 * snooker_alpha
                
                # CVXPY: minimize || rot_error - J_rot @ dqa ||^2
                rot_cost = rotation_tracking_weight * cp.sum_squares(
                    cp.Constant(rot_error) - cp.Constant(J_rot) @ dqa
                )
                obj_terms.append(rot_cost)
                obj_names.append("global_rotation_tracking")
                global_rotation_tracking_added = True
                
                # 详细调试信息
                if (frame_idx == 0) or (self.debug and frame_idx % 100 == 0):
                    rot_error_deg = np.rad2deg(np.linalg.norm(rot_error))
                    debug_msg = (
                        f"\n=== [Global Rotation Tracking - Frame {frame_idx}] ===\n"
                        f"  snooker_alpha: {snooker_alpha:.4f}\n"
                        f"  rotation_tracking_weight: {rotation_tracking_weight:.4f}\n"
                        f"  target_quat (wxyz): [{target_lh_quat[0]:.4f}, {target_lh_quat[1]:.4f}, {target_lh_quat[2]:.4f}, {target_lh_quat[3]:.4f}]\n"
                        f"  current_quat (wxyz): [{current_quat[0]:.4f}, {current_quat[1]:.4f}, {current_quat[2]:.4f}, {current_quat[3]:.4f}]\n"
                        f"  rot_error (rotvec): [{rot_error[0]:.4f}, {rot_error[1]:.4f}, {rot_error[2]:.4f}]\n"
                        f"  rot_error magnitude: {rot_error_deg:.2f} deg\n"
                        f"  J_rot shape: {J_rot.shape}\n"
                        f"  expected rotation cost: {rotation_tracking_weight * np.sum(rot_error**2):.6f}\n"
                    )
                    print(debug_msg)
                    with open(self.log_path, "a") as f:
                        f.write(debug_msg)
                        
            except Exception as e:
                if frame_idx == 0:
                    print(f"[WARNING] Global rotation tracking failed: {e}")
                    with open(self.log_path, "a") as f:
                        f.write(f"\n[WARNING Frame {frame_idx}] Global rotation tracking failed: {e}\n")

        lap_term = cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec))
        obj_terms.append(lap_term)
        obj_names.append("laplacian")

        # nominal tracking for selected indices
        # ! cube: 逻辑解耦 —— 现在由 activate_general_nominal_tracking 控制是否进行“全身其他关节”的追踪
        if self.activate_general_nominal_tracking and is_full_nominal and (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])
                term = w_nominal_tracking * cp.sum_squares(z)
                obj_terms.append(term)
                obj_names.append("general_nominal")

        # Q_diag cost
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        q_diag_term = cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last))
        obj_terms.append(q_diag_term)
        obj_names.append("q_diag")

        # Smoothness cost
        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        if np.isscalar(self.smooth_weight):
            smooth_term = self.smooth_weight * cp.sum_squares(dqa - dqa_smooth)
            obj_terms.append(smooth_term)
        else:
            Wsmooth = np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                smooth_term = cp.sum_squares(cp.multiply(np.sqrt(Wsmooth), dqa - dqa_smooth))
                obj_terms.append(smooth_term)
            else:
                smooth_term = cp.quad_form(dqa - dqa_smooth, Wsmooth)
                obj_terms.append(smooth_term)
        obj_names.append("smoothness")

        # obj_terms 是一个列表，包含了所有的成本分量（Laplacian、Tracking、Smoothness 等）。
        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        # -------- Solve with Clarabel --------
        solver_kwargs = {"verbose": verbose}
        problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        
        # PRINT调试信息：如果 Cost 异常（例如大于 100），打印分量
        if frame_idx % 3 == 0:
             print(f"\n[Debug Frame {frame_idx}] Total Cost: {problem.value:.2f}")
             for name, term in zip(obj_names, obj_terms):
                 print(f"  - {name}: {term.value:.2f}")
        if (problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) and init_t:
            constraints = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solve failed: {problem.status}")

        dqa_star = dqa.value
        cost = problem.value

        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa_star + q_a_n_last
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12

        return q_star, cost

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,
        frame_idx: int = 0,
        is_full_nominal: bool = False,
        target_lh_quat: np.ndarray | None = None,  #! wrist: 新增 - 目标全局四元数
    ):
        """Iterate the solver for multiple iterations."""
        last_cost = np.inf
        for _ in range(n_iter):
            q_a_n_last = q_n[self.q_a_indices]
            q_n, cost = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_a_n_last,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
                frame_idx=frame_idx,
                is_full_nominal=is_full_nominal,
                target_lh_quat=target_lh_quat,  #! wrist: 新增 - 传递目标全局四元数
            )
            if np.isclose(cost, last_cost):
                break
            last_cost = cost
        return q_n, cost

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        # Update robot joint configurations
        robot_joint_positions = q[7 : 7 + self.task_constants.ROBOT_DOF]
        self.viser_robot.update_cfg(robot_joint_positions)

        # Update robot base pose using set_transform
        robot_quat = q[3:7]  # Base orientation
        robot_pos = q[:3]  # Base position

        # Update robot base frame
        self.robot_base.position = robot_pos
        self.robot_base.wxyz = robot_quat  # Assuming quaternion is in wxyz order

        # Update object pose if it exists
        if hasattr(self, "viser_object") and self.viser_object is not None:
            if self.has_dynamic_object:
                object_quat = q[-4:]
                object_pos = q[-7:-4]
            else:
                object_quat = np.asarray([1, 0, 0, 0])
                object_pos = np.zeros(3)

            # Update object base frame
            self.object_base.position = object_pos
            self.object_base.wxyz = object_quat  # Assuming quaternion is in wxyz order

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1)):
        """Draw keypoints in visualization."""
        if not hasattr(self, "server"):
            return None

        # Create a sphere mesh using trimesh
        sphere = trimesh.primitives.Sphere(radius=0.02)
        vertices = sphere.vertices
        faces = sphere.faces

        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])

        kpts_handle_list = []

        # Draw keypoints
        if len(p.shape) == 1:
            # Single point
            kpts_handle = self.server.scene.add_mesh_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                position=p,
                color=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)
        elif len(p.shape) == 2:
            # Multiple points
            kpts_handle = self.server.scene.add_batched_meshes_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                batched_positions=p,
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
                batched_colors=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)

        return kpts_handle_list

    def visualize_motion(
        self,
        human_joint_motions,
        obj_pts_demo,
        obj_pts,
        retargeted_motions,
        tetrahedra,
        dt=1 / 30,
        visualize_tetrahedra=False,
    ):
        for i in range(len(human_joint_motions)):
            object_pts_demo = obj_pts_demo[i]
            object_pts = obj_pts[i]
            self.draw_keypoints(human_joint_motions[i, self.smplh_mapped_joint_indices], name="human")
            self.draw_keypoints(object_pts_demo, name="object_demo", rgba=(1, 0, 0, 1))
            self.draw_keypoints(object_pts, name="object", rgba=(0, 1, 0, 1))
            self.draw_q(retargeted_motions[i])
            robot_link_positions = self._get_robot_link_positions(
                retargeted_motions[i], self.laplacian_match_links.values()
            )
            self.draw_keypoints(robot_link_positions, name="robot", rgba=(0, 1, 0, 1))
            input()
            if visualize_tetrahedra:
                self.visualize_tetrahedra(
                    np.vstack(
                        [
                            human_joint_motions[i, self.smplh_mapped_joint_indices],
                            object_pts_demo,
                        ]
                    ),
                    tetrahedra[i],
                    name="human_tetrahedra",
                )
                self.visualize_tetrahedra(
                    np.vstack([robot_link_positions, object_pts]),
                    tetrahedra[i],
                    name="robot_tetrahedra",
                    rgba=(0, 1, 1, 1),
                )
            else:
                time.sleep(dt)

    def visualize_tetrahedra(self, vertices, tetrahedra, name="tetrahedra", color=(0, 0, 0, 1)):
        # Convert color to 0-255 range
        color_255 = np.array(color[:3]) * 255

        # Prepare points and colors for all edges
        points = []
        colors = []

        for tet in tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = tet[i], tet[j]
                    points.extend([vertices[u], vertices[v]])
                    colors.extend([color_255, color_255])

        # Convert to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Add line segments for all edges at once
        self.server.scene.add_line_segments(
            f"/{name}",
            points=points,
            colors=colors,
            line_width=0.01,
        )

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        J_bodyA = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            if self.object_name in self._geom_names[g1] and "ground" in self._geom_names[g2]:
                return False
            if "ground" in self._geom_names[g1] and self.object_name in self._geom_names[g2]:
                return False
            return (
                self.object_name in self._geom_names[g1]
                or self.object_name in self._geom_names[g2]
                or "ground" in self._geom_names[g1]
                or "ground" in self._geom_names[g2]
            )

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(g1), m.geom(g2), self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    def _world_to_body_frame(self, p_w: np.ndarray, body_idx: int) -> np.ndarray:
        """Transform point from world frame to body frame."""
        p_w = np.asarray(p_w).reshape(3)
        body_pos = self.robot_data.xpos[body_idx].reshape(3)
        body_mat = self.robot_data.xmat[body_idx].reshape(3, 3)
        return body_mat.T @ (p_w - body_pos)

    def _get_geometry_name(self, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE
        qadr = self.robot_model.jnt_qposadr[j0]  # 0
        dadr = self.robot_model.jnt_dofadr[j0]  # 0

        # Linear block: v_lin = xyz_dot
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

        # Angular block: ω_* = 2 * E_*(q) * quat_dot
        w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        # ---- FREE joint #1 (human/root): use model addresses, but this should be the first joint ----
        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])  # expect 0
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])  # start of its 6 qvel dofs

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3 : qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
        T[dadr1 + 0 : dadr1 + 3, qadr1 + 0 : qadr1 + 3] = np.eye(3)  # v block
        T[dadr1 + 3 : dadr1 + 6, qadr1 + 3 : qadr1 + 7] = E1  # ω block

        if self.has_dynamic_object:
            # ---- FREE joint #2 (object): assume it's the last FREE joint; fill its 6x7 block ----
            # Find it by type (safer than hardcoding tail indices)
            free_joints = [
                j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (human + object)."
            j_free2 = free_joints[1]  # second FREE joint
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])  # expect nq-7
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])  # its 6 qvel dofs (often at nv-6)

            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3 : qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0 : dadr2 + 3, qadr2 + 0 : qadr2 + 3] = np.eye(3)  # v block
            T[dadr2 + 3 : dadr2 + 6, qadr2 + 3 : qadr2 + 7] = E2  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    #! wrist
    def _calc_rotation_jacobian_and_error(
        self,
        q: np.ndarray,
        link_name: str,
        target_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算机器人 link 的旋转雅可比和与目标旋转的误差。
        
        这是一个模块化的新增函数，用于支持全局旋转 tracking。
        
        Args:
            q: 当前机器人配置 (nq,)
            link_name: 目标 link 名称 (e.g., "left_wrist_yaw_link")
            target_quat_wxyz: 目标全局旋转四元数 [w, x, y, z]
        
        Returns:
            J_rot: (3, nq_a) 旋转雅可比矩阵（将 dqa 映射到角速度）
            rot_error: (3,) 旋转误差向量（轴角/旋转向量表示）
            current_quat_wxyz: (4,) 当前全局四元数 [w, x, y, z]，用于调试
        """
        # 1. 更新正向运动学
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        
        # 2. 获取 body id 和当前全局旋转
        body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if body_id == -1:
            raise ValueError(f"Body '{link_name}' not found in MuJoCo model")
        
        current_rot_mat = self.robot_data.xmat[body_id].reshape(3, 3)
        pos = self.robot_data.xpos[body_id]
        
        # 3. 计算旋转雅可比（使用 mj_jac 获取 Jr）
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")  # 旋转雅可比
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, pos.reshape(3, 1), body_id)
        
        # 4. 转换到 qpos 空间（使用 T 矩阵）
        T = self._build_transform_qdot_to_qvel_fast()
        J_rot_full = Jr @ T  # (3, nq)
        J_rot = J_rot_full[:, self.q_a_indices]  # (3, nq_a)
        
        # 5. 计算旋转误差
        # 目标旋转 (从 wxyz 转为 scipy 的 xyzw)
        target_rot = Rotation.from_quat([
            target_quat_wxyz[1], target_quat_wxyz[2], 
            target_quat_wxyz[3], target_quat_wxyz[0]
        ])
        # 当前旋转
        current_rot = Rotation.from_matrix(current_rot_mat)
        
        # 误差 = target * current^(-1)，表示从当前到目标需要的旋转
        rot_error_obj = target_rot * current_rot.inv()
        
        # 转为轴角表示，取旋转向量作为误差 (方向 = 旋转轴, 模长 = 旋转角度 rad)
        rot_error = rot_error_obj.as_rotvec()  # (3,)
        
        # 返回当前四元数用于调试 (转回 wxyz)
        current_quat_xyzw = current_rot.as_quat()  # scipy 返回 xyzw
        current_quat_wxyz = np.array([
            current_quat_xyzw[3], current_quat_xyzw[0], 
            current_quat_xyzw[1], current_quat_xyzw[2]
        ])
        
        return J_rot, rot_error, current_quat_wxyz

    
    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position-based Jacobians using MuJoCo."""
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)

        q_mujoco = q.copy()
        self.robot_data.qpos[:] = q_mujoco

        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

            if point_offsets is not None:
                pC_B = point_offsets
            else:
                # pC_B = np.zeros(3) #! cube
                #! cube: 优先级：task_constants 中的偏移 > 本文件 snooker 虚拟点偏移 > 全零
                virtual_offsets = {}
                if hasattr(self.task_constants, "VIRTUAL_SITE_OFFSETS"):
                    virtual_offsets.update(self.task_constants.VIRTUAL_SITE_OFFSETS)
                virtual_offsets.update(self.virtual_site_offsets)
                pC_B = np.array(virtual_offsets.get(name, [0.0, 0.0, 0.0]))

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            # Store reduced Jacobian and position with hard copies to avoid aliasing
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)  # FIX (copy)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None

        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco."""
        mujoco_q = q.copy()

        # Set the configuration
        if mujoco_q.shape != self.robot_data.qpos.shape:
            self.robot_data.qpos = mujoco_q[:-7]  # Exclude object information from q
        else:
            self.robot_data.qpos = mujoco_q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []

        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)
