
# GMR & Holosoma 重定向原理深度分析


## 1. 逐帧处理 vs. 序列处理*   

**GMR (Online-retargeting)**:    
*   **本质**: 纯粹的**逐帧优化 (Frame-by-frame IK)**。    
*   **实现**: 在 `motion_retarget.py` 的 `retarget` 方法中，每输入一帧人体数据 `human_data`，它通过 `mink.solve_ik` 调用底层求解器，计算出当前帧的最佳 `qpos`。每一帧的求解相对独立（仅以当前姿态作为初值）。*   

**Holosoma**:    
*   **本质**: 同样是**逐帧执行的序列优化**。    
*   **实现**: 在 `interaction_mesh_retargeter.py` 的 `retarget_motion` 方法中，它遍历整个动作序列 (`for i in pbar`)，但每一帧都会调用 `solve_single_iteration` 解决一个二次规划 (QP) 问题。    
*   **区别**: 虽然是逐帧求解，但 Holosoma 在目标函数中加入了**平滑项 (Smoothness cost)** 和针对时间步的缩放策略（如 `w_nominal_tracking` 随时间衰减），使其在序列上表现更连贯。

## 2. 目标函数 (Objective Function) 的核心差异

### GMR 的目标函数：姿态追踪 (Pose Tracking)
*  **核心逻辑**: 最小化**世界坐标系下**的偏差。
*   **代码参考**: `motion_retarget.py` 中的 `FrameTask`。
*   **数学本质**:     \[ \min \| p_{robot} - p_{human, target} \|^2 + \| \text{rot}_{robot} - \text{rot}_{human, target} \|^2 \]    它将机器人关节强行推向人体关节在世界坐标系中的对应位置。如果身材比例不一致（如人手长、机器人手短），这种方法容易导致肢体扭曲或无法触及。

### Holosoma 的目标函数：交互网格 (Interaction Mesh)
*   **核心逻辑**: 最小化**局部微分坐标 (Laplacian Coordinates)** 的偏差。
*   **代码参考**: `interaction_mesh_retargeter.py` 中的 `obj_terms.append(cp.sum_squares(lap_var - target_lap_vec))`。
*   **数学本质**:    1.  **Laplacian 项**: 维护的是**相对关系**。它计算每个关节与其邻居（其他关节或物体点）的向量差。这意味着它不在乎关节在世界坐标系的绝对位置，而在乎“手离躯干多远”、“两手之间距离是多少”。    2.  **Nominal Tracking 项**: 这部分确实包含**世界坐标系下的位置误差最小化**（对应 `q_a_nominal`），但它通常权重较低，仅作为参考姿态。    3.  **约束项**: 包含严苛的非穿透约束 (Non-penetration) 和关节限位。

## 3. 无物体 (Robot-only) 模式下的差异

如果在重定向中没有涉及 Object（如 `robot_only` 任务），两者的目标函数**依然有显著区别**：

1.  **参考系不同**:     
*   **GMR** 依然尝试让机器人关节去“够”世界坐标系里的目标点。    
*   **Holosoma** 会在后台创建一个“地面 (Ground)”作为默认 Object（参考 `interaction_mesh_retargeter.py` 第 347 行）。它会将人体所有关节与地面点建立 Delaunay 三角剖分网格。

2.  **保形性 (Shape Preservation)**:     
*   即使没有物体，Holosoma 的 Laplacian 约束也会强制机器人保持与人体类似的“姿态拓扑”。例如，如果人弯腰，Holosoma 维护的是脊柱与腿、地面的相对角度和距离，而不是简单地追踪脊柱的世界坐标。

3.  **身材适配**:    
*   由于 Holosoma 优化的是相对距离比例，它在处理身材差异（如 G1 机器人与成年男性）时，比 GMR 的绝对坐标追踪更不容易出现“拉伸感”或“扭曲感”。