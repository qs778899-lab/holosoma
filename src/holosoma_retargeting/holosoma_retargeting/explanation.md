
# GMR 与 Holosoma 重定向原理深度分析


## 1. 逐帧处理 vs. 序列处理

*   **GMR (Online-retargeting)**:
    *   **本质**: 单纯的**逐帧优化 (Frame-by-frame IK)**。
    *   **实现**: 在 `motion_retarget.py` 的 `retarget` 方法中，每输入一帧人体数据 `human_data`，它通过 `mink.solve_ik` 调用底层求解器，计算出当前帧的最佳 `qpos`。每一帧的求解相对独立（仅以当前姿态作为初值）。
*   **Holosoma (Offline-retargeting)**:
    *   **本质**: 同样是**逐帧执行的序列优化**。
    *   **实现**:  在 `interaction_mesh_retargeter.py` 的 `retarget_motion` 方法中，它通过一个循环按时间顺序逐帧处理序列。
    *   **区别**: 虽然是逐帧求解，但 Holosoma 在目标函数中加入了**平滑项 (Smoothness cost)**（代码第 612-621 行）和针对时间步的缩放策略（初始化阶段w_nominal_tracking权重高先靠近标准姿态， 后随着时间推移权重下降让机器人能够更自由地跟随 Laplacian 交互网格 进行运动），使其在序列上表现更连贯。

## 2. 目标函数 (Objective Function) 的核心差异

### GMR 的目标函数：姿态追踪 (Pose Tracking)
*   **核心逻辑**: 最小化**世界坐标系下**的偏差。
*   **代码参考**: `motion_retarget.py` 中的 `FrameTask`。
*   **数学本质**: 
    $$ \min \| p_{robot} - p_{human, target} \|^2 + \| \text{rot}_{robot} - \text{rot}_{human, target} \|^2 $$
    它将机器人关节强行推向人体关节在世界坐标系中的对应位置。如果身材比例不一致（如人手长、机器人手短），这种方法容易导致肢体扭曲或无法触及。
    *   **G1 关节选择**: 
        基于 `ik_configs/bvh_nokov_to_g1.json`。GMR 为 G1 显式定义了 **14 个核心追踪连杆**：
        *   **躯干与基座**: `pelvis` (盆骨), `torso_link` (对应人体 Spine2)。
        *   **下肢 (双侧)**: `hip_yaw_link`, `knee_link`, `ankle_roll_link` (足部)。
        *   **上肢 (双侧)**: `shoulder_yaw_link`, `elbow_link`, `wrist_yaw_link` (手腕)。

        为什么wrist只选择yaw一个自由度：链条顺序：肘部 → wrist_roll → wrist_pitch → wrist_yaw → 橡胶手(固定)。在逆运动学中，如果你想控制整只手的最终姿态（位置和旋转），你必须把目标（Constraint Target）设置在这条链条的最后一个连杆上。

### Holosoma 的目标函数：交互网格 (Interaction Mesh)
*   **核心逻辑**: 最小化**局部微分坐标 (Laplacian Coordinates)** 的偏差。
*   **代码参考**: `interaction_mesh_retargeter.py` 中的 `solve_single_iteration` 方法。
*   **详细补充**:
    1.  **Laplacian 项 (核心成本)**: 
        *   **代码实现**: `obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))` (第 598 行)。
        *   **细节**: 它计算关节 \(V_i\) 与其所有邻居 \(V_j\) 的相对向量之和（即 \(L \cdot V\)）。邻居关系通过 Delaunay 三角化确定（第 354 行）。这使得算法优化的不是“手在世界坐标系的 (x,y,z)”，而是“手相对于躯干、膝盖以及物体的**相对位移向量**”。
        *   **G1 关节选择**: 基于 `data_type.py` 中的 `JOINTS_MAPPINGS`。对于 G1 机器人，包含全身核心的 **13 个关键点**（髋部，左右大腿，左右膝盖，左右大臂，左右小臂，左右脚踝，左右手）。
    2.  **Nominal Tracking 项 (姿态参考)**: 
        *   **代码实现**: `obj_terms.append(w_nominal_tracking * cp.sum_squares(z))` (第 605 行)。`z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])` (第 604 行)。
        `z` 代表当前关节角速度与目标关节角速度的偏差。
        *   **原理解析**: 虽然代码中处理的是 `dqa`（角度增量/速
        度），但其目标是最小化“实际运动量”与“到达目标位置所需运动
        量”之间的差距。这是一种在**速度空间执行的位置追踪**，可以
        减少动作的不连续。
        *   **G1 关节选择**: 基于 `robot.py` 中的 `NOMINAL_TRACKING_INDICES`。对于 G1 机器人，仅包含**前 19 个自由度**（根节点位姿 + 12 个腿部驱动关节）。它决定了“下半身站得稳不稳”。
    3.  **约束项 (硬约束)**: 
        *   **代码实现**: 包括关节限位 (第 586 行)、足部锁定 (第 551 行) 和基于 MuJoCo 碰撞检测的非穿透约束 (第 578 行)。这些是必须满足的硬条件，优先级高于 Laplacian 成本。

## 3. 无物体 (Robot-only) 模式下的差异

如果在重定向中没有涉及 Object（如 `robot_only` 任务），两者的目标函数**依然有显著区别**：

1.  **参考系不同**: 
    *   **GMR** 依然尝试让机器人关节去“够”世界坐标系里的目标点。
    *   **Holosoma** 的逻辑如下：
        *   **代码实现**: `if self.object_name == "ground": human_mapped_joints_in_object = human_mapped_joints` (第 347 行)。
        *   **具体行为**: 即使没有显式物体，Holosoma 也会将人体关键关节与一组虚拟的“地面采样点”连接（在 `robot_retarget.py` 的 `setup_object_data` 中通过 `create_ground_points` 生成）。
        *   **后果**: 这意味着机器人的每一个姿态都是**相对于地面高度和范围**进行解算的。这天然地防止了“浮空步”或“深蹲过度”，因为 Laplacian 算子会努力维持脚与地面的相对网格关系。

2.  **身材适配**:    
*   由于 Holosoma 优化的是相对距离比例，它在处理身材差异（如 G1 机器人与成年男性）时，比 GMR 的绝对坐标追踪更不容易出现“拉伸感”或“扭曲感”。

## 4. 数学框架：二次规划 (Quadratic Programming, QP)

### 共同点
GMR 和 Holosoma 在本质上都属于**基于优化（Optimization-based）的逆运动学**。它们每一帧都在解一个二次规划问题：

$$ \min_{\Delta q} \frac{1}{2} \Delta q^T P \Delta q + q^T \Delta q $$
$$ \text{s.t. } A \Delta q \le b $$

一个 QP 问题 的标准形式是：
- **目标函数**：是变量的二次方。例如最小化误差的平方：$\min \| \text{误差} \|^2$。
- **约束条件**：是变量的一次方（线性）。例如关节限位：$q_{\min} \le q + \Delta q \le q_{\max}$。

### 实现方式的区别
1.  **GMR (封装型 QP)**:
    *   使用 `mink` 库。用户定义“任务 (Task)”，库自动将任务转化为 QP 矩阵。
    *   **优点**: 代码简洁，适合标准的坐标追踪。
    *   **缺点**: 难以定制复杂的局部拓扑约束。

2.  **Holosoma (原生型 QP)**:
    *   使用 `cvxpy` 库。开发者手动构建目标函数（第 598 行的 `lap_var`）和约束矩阵。
    *   **优点**: 极高的自定义能力。Holosoma 借此实现了“交互网格（Interaction Mesh）”，即不仅仅追踪关节的位置，还追踪关节之间的网格形变量。
    *   **缺点**: 代码实现复杂，需要深厚的数学基础来手动推导雅可比矩阵 (Jacobians)。

## 5. 实际使用体验

    *   holosoma运行慢很多
    *   GMR得到的robot腿部的运动更自然
    *   holosoma腿部运动容易内八的原因：（1）在 Laplacian项中，只考虑点对点，无角度关系。 (2) Nominal 项的权重太低。



## 6. 补充和改进：

1. Joint 和 Link: 
    
    * Link（连杆）: 机器人的刚性身体部分, 在retarget中将人的关节映射到link。
    * Joint（关节）：连接两个 Link 的部件，在robot中的主动关节（DOF）指这些旋转轴。

    输入（点）：人类的关节坐标。

    计算（优化）：寻找一组 Joint 角度。

    结果（Link）：使得机器人的 Link 上的特定点与人类的关节坐标尽量重合。

2. 改进腿部内八的不自然姿态：

    Note: 不同bvh文件的left toe offset可能不同，但同一bvh文件肯定相同。

    bvh文件分为 HIERARCHY 和 MOTION 两大部分。（运行 python check_bvh_alignment.py snooker/snooker2.bvh 可查看具体数据格式和内容）
    - **HIERARCHY 部分**是一个分层树结构，定义了骨骼的拓扑关系。每一个 Joint 记录了相对于父节点的 `OFFSET`（三维偏移量），这决定了骨骼的静态比例（如大腿长度）。同时，它通过 `CHANNELS` 声明了该关节在 MOTION 部分占用的数据量和含义。
    - **MOTION 部分**是动画序列的原始数值记录。每一行代表一帧（Frame），包含了一长串由空格分隔的数字。这些数字的排列顺序严格遵循 HIERARCHY 部分定义的深度优先遍历顺序。例如索引0-5是Hips (Root)的Xpos, Ypos, Zpos, Yrot, Xrot, Zrot。
    - **两者关系**：MOTION 部分的数字本身没有标签，必须依靠 HIERARCHY 定义的通道数（如 Hips 占 6 个位移+旋转通道，普通关节占 3 个旋转通道）来依次解析。
    
    （所以，Nominal 项也无法track脚踝的角度?  那GMR是在怎么处理这种bvh文件，是否有内置库函数可以先计算出每个joint的旋转角）

    bvh文件中LeftFoot的特殊完整数据内容：
    JOINT LeftFoot
      {
      OFFSET 0.0 -42.057999 0.0 #脚踝相对于膝盖的偏移向量
      CHANNELS 3 Yrotation Xrotation Zrotation
      End Site
      {
      OFFSET 0.000000 -10.000000 15.120000 #脚尖相对于脚踝的偏移向量
      }
      }


3. urdf 和 xml 模型文件格式

    * URDF：主要用于描述单个机器人的运动学结构（树状结构，只有一个 Root）。

    * XML：专为物理仿真设计。它可以定义整个“世界”，包括多个独立的物体、复杂的碰撞、真实的光照和材质。

4. G1 link列表

    | 连杆名称 (Link Name) | 中文名称 | 备注 |
    | :--- | :--- | :--- |
    | **pelvis** | 盆骨 | 机器人的根部基座 (Root) |
    | **waist_yaw/roll_link** | 腰部偏航/横滚连杆 | 控制腰部转动的环节 |
    | **torso_link** | 躯干连杆 | 机器人的上半身主体 |
    | **head_link** | 头部连杆 | 头部位置 |
    | **left/right_hip_pitch_link** | 髋部俯仰连杆 | 大腿根部，控制腿部前后摆动 |
    | **left/right_hip_roll_link** | 髋部横滚连杆 | 大腿根部，控制腿部左右摆动 |
    | **left/right_hip_yaw_link** | 髋部偏航连杆 | 大腿根部，控制腿部旋转 |
    | **left/right_knee_link** | 膝部连杆 | 膝关节 |
    | **left/right_ankle_pitch_link**| 踝部俯仰连杆 | 踝关节前后移动 |
    | **left/right_ankle_roll_link** | 踝部横滚连杆 | 踝关节左右倾斜 |
    | **left/right_shoulder_pitch_link**| 肩部俯仰连杆 | 肩膀前后摆动 |
    | **left/right_shoulder_roll_link** | 肩部横滚连杆 | 肩膀左右摆动 |
    | **left/right_shoulder_yaw_link** | 肩部偏航连杆 | 大臂旋转 |
    | **left/right_elbow_link** | 肘部连杆 | 肘关节 |
    | **left/right_wrist_roll_link** | 腕部横滚连杆 | 手腕旋转 |
    | **left/right_wrist_pitch_link** | 腕部俯仰连杆 | 手腕上下摆动 |
    | **left/right_wrist_yaw_link** | 腕部偏航连杆 | 手腕左右转动 |
    | **left/right_rubber_hand** | 橡胶手 | 机器人的末端手部 (无手指版本) |
    | **left/right_toe_link** |  脚尖连杆



    