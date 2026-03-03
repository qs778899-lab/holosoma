"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import tyro


# 使用 FlagConversionOff 让 bool 参数接受显式的 True/False 值
BoolWithExplicitValue = Annotated[bool, tyro.conf.FlagConversionOff]


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    collision_detection_threshold: float = 0.1
    """Large-scale distance threshold (meters) to start detecting any collisions."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""

    snooker_frame_range: list[int] | None = None
    """[start_frame, end_frame] where snooker constraints are active.
    This is a legacy parameter that controls both Laplacian and wrist tracking.
    Use laplacian_frame_range and wrist_tracking_frame_range for independent control."""

    laplacian_frame_range: list[int] | None = None
    """[start_frame, end_frame] where Laplacian virtual points (CueTip, LeftHandBridge, RightHandGrip) are active.
    If None, falls back to snooker_frame_range."""

    wrist_tracking_frame_range: list[int] | None = None
    """[start_frame, end_frame] where wrist rotation tracking is active.
    If None, falls back to snooker_frame_range."""

    activate_snooker_tracking: BoolWithExplicitValue = False
    """Whether to enable left wrist yaw tracking for snooker task."""

    activate_snooker_laplacian: BoolWithExplicitValue = False
    """Whether to add snooker virtual points (CueTip, LeftHandBridge, RightHandGrip) to Laplacian mesh."""

    activate_realtime_rotation_tracking: BoolWithExplicitValue = False
    """Whether to extract rotation from 7D data for nominal tracking (when no reference npz)."""

    activate_general_nominal_tracking: BoolWithExplicitValue = False
    """Whether to enable full-body nominal tracking (requires reference sequence)."""

    activate_palm_flat_constraint: BoolWithExplicitValue = False
    """Whether to enable left palm flat orientation constraint."""

    visualization_fps: int = 30
    """Initial FPS for visualization playback. Lower values reduce blur but may appear less smooth."""

    visualization_interp_mult: int = 2
    """Visual FPS multiplier for interpolation. Set to 1 to disable interpolation and reduce blur.
    Higher values create smoother motion but may cause visual accumulation blur over time."""

    smooth_weight: float = 0.2
    """Weight for smoothness cost that penalizes changes between consecutive frames.
    Higher values (e.g., 10.0) enforce stronger smoothness, reducing jitter but potentially making motion less responsive.
    Lower values (e.g., 0.2) allow more frame-to-frame variation."""

    activate_right_wrist_yaw_zero_constraint: BoolWithExplicitValue = False
    """Whether to enable right wrist yaw zeroing soft constraint."""

    right_wrist_yaw_zero_frame_range: list[int] | None = None
    """[start_frame, end_frame] where right wrist yaw zeroing is active.
    Default: [580, 1300] if None."""

    right_wrist_yaw_zero_ramp_frames: int = 200
    """Number of frames for smooth transition (ramp-up/down) for right wrist yaw zeroing."""

    right_wrist_yaw_zero_weight: float = 10.0
    """Weight for right wrist yaw zeroing soft constraint."""

    virtual_pos_frame_range: list[int] | None = None
    """[start_frame, end_frame] where human virtual position height compensation is active.
    Default: [250, 1300] if None."""

    virtual_pos_ramp_frames: int = 20
    """Number of frames for smooth transition (ramp-up/down) for virtual position height compensation."""

    virtual_pos_target_z: float = 0.7
    """Target Z height for human virtual position height compensation."""

    activate_foot_leg_weight_boost: BoolWithExplicitValue = False
    """Whether to boost weights for feet and legs in Laplacian mesh."""

    foot_leg_boost_weight: float = 20.0
    """Target boosted weight for feet and legs."""

    foot_leg_boost_frame_range: list[int] | None = None
    """[start_frame, end_frame] where foot and leg weight boost is active."""

    foot_leg_boost_ramp_frames: int = 50
    """Number of frames for smooth transition (ramp-up/down) for weight boost."""

    leg_self_collision_margin: float = 0.0
    """Safety margin (minimum distance in meters) for leg-to-leg self-collision."""

    leg_self_collision_detection_threshold: float = 0.02
    """Distance threshold (meters) to activate cross-side leg self-collision constraints."""

    activate_foot_xy_tracking: BoolWithExplicitValue = False
    """Whether to enable foot XY absolute position tracking soft constraint."""

    foot_xy_tracking_weight: float = 10.0
    """Weight for foot XY absolute position tracking soft constraint."""

    foot_xy_tracking_frame_range: list[int] | None = None
    """[start_frame, end_frame] where foot XY tracking is active."""

    foot_xy_tracking_ramp_frames: int = 50
    """Number of frames for smooth transition (ramp-up/down) for foot XY tracking."""
