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
