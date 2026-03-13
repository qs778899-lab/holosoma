
# 基于交互网格的重定向方法数学原理

本文档结合 `holosoma_retargeting` 代码实现，分析了基于交互网格（Interaction Mesh）的 Laplacian 坐标重定向方法的数学原理。

## 1. Laplacian 坐标定义

第 $i$ 个关键点 $p_{t,i} \in P_t$ 的 Laplacian 坐标定义为该点与其邻域 $N(i)$ 内所有邻居点 $j$ 的加权平均值之差（其中 $t$ 表示当前时间步或帧）：

$$L(p_{t,i}) = p_{t,i} - \sum_{j \in N(i)} w_{ij} \cdot p_{t,j} \quad (1)$$

其中 $w_{ij}$ 是归一化的权重，$j$ 为邻居节点的索引。在本项目中，默认使用均匀权重（Uniform Weights），即：
$$w_{ij} = \frac{1}{|N(i)|}$$

### 几何意义
Laplacian 坐标不再关注点的绝对世界坐标，而是关注点的**局部几何特征**：
1. **相对位置描述**：它描述了点 $p_{t,i}$ 相对于其邻域中心（质心）的偏移向量。
2. **形状保持**：在重定向过程中，保持 Laplacian 坐标不变，意味着保持了点与点之间的相对空间关系（如手与物体的距离、手臂的弯曲程度），从而实现“形散而神不散”的动作迁移。
3. **曲率近似**：在几何处理中，Laplacian 坐标的方向通常近似于该点处的法线方向，其模长正比于该点处的平均曲率。它捕捉了局部形状偏离平坦程度的大小。

### 代码实现
在 `src/holosoma_retargeting/holosoma_retargeting/src/utils.py` 中，`calculate_laplacian_matrix` 函数实现了这一逻辑：

```python
# 均匀权重实现 (utils.py)
if uniform_weight:
    weights = np.ones(len(neighbors_indices)) / len(neighbors_indices)
# ...
laplacian_matrix[i, i] = 1.0
for j, neighbor_idx in enumerate(neighbors_indices):
    laplacian_matrix[i, neighbor_idx] = -weights[j]
```

该矩阵 $L$ 满足 $\mathcal{L} \mathbf{V} = \mathbf{\Delta}$，其中 $\mathbf{V}$ 是顶点坐标，$\mathbf{\Delta}$ 是得到的 Laplacian 坐标向量。

## 2. 优化目标函数 (Objective Function)

重定向的核心是一个带约束的二次规划（QP）问题。其目标函数 $E_{total}$ 由多个加权项组成，旨在平衡几何特征保持、运动平滑性和特定任务约束。

### 优化目标函数
$$E_{total} = w_L E_L + w_{smooth} E_{smooth} + w_{diag} E_{diag}$$

### 各分量解释

1. **拉普拉斯变形能量项 ($E_L$)**：
   使目标网格 $P_{t,target}$ 的 拉普拉斯坐标尽可能接近源演示网格 $P_{t,source}$ 的 拉普拉斯坐标：
   $$E_L = \sum_{i} \|L(p^{source}_{t,i}) - L(p^{target}_{t,i})\|^2 \quad $$


2. **运动平滑项 ($E_{smooth}$)**：
   惩罚相邻时间步之间的关节速度变化，确保动作连贯，减少高频抖动：
   $$E_{smooth} = \| \Delta q_a - \Delta q_{a, last} \|^2$$ 
   $\Delta q_a$ 是当前待求解的优化变量，$\Delta q_{a, last}$ 是上一帧已知的位移常量。

3. **正则化项 ($E_{diag}$)**：
   对关节位移增量进行选择性惩罚，主要用于解决机器人的冗余自由度问题。在有多个可行解时，引导机器人选择更接近标称姿态的解：
   $$E_{diag} = \| \sqrt{Q_d} \cdot (q_{a, last} + \Delta q_a) \|^2$$
    $Q_d$ 是一个对角权重矩阵。

## 3. 硬约束 (Hard Constraints)

硬约束是优化问题中必须严格满足的条件，如果无法满足，求解器将报错或进入降级逻辑（Fallback）。

### 运动学与物理约束

1. **关节限位约束**：
   确保求解出的关节角度在机器人的物理极限范围内：
   $$q_{min} \leq q_{a, last} + \Delta q_a \leq q_{max}$$

2. **信赖域约束**：
   限制单步迭代的最大位移，确保线性化近似的有效性：
   $$\| \Delta q_a \|_2 \leq \text{step\_size}$$
   在代码中通过二阶锥约束 `cp.SOC` 实现。

### 3.2 接触与碰撞约束
1. **足端固定约束**：
   当检测到脚部处于支撑相时，约束其在 XY 平面的位移为 0，防止“滑步”：
   $$J_{foot, xy} \cdot \Delta q_a \approx 0$$

2. **防穿透与自碰撞约束**：
   防止机器人与地面、物体发生穿透，以及机器人自身的肢体碰撞。
   通过计算几何体间的距离 $\phi$ 及其雅可比 $J_{coll}$ 实现：
   $$J_{coll} \cdot \Delta q_a \geq \text{margin} - \phi$$


---

## 4. 差分逆运动学 (DiffIK) 结合

为了在机器人上实现该重定向，系统将 Laplacian 坐标的变化与机器人的关节速度（或位置增量 $\Delta q_a$）联系起来。

### 线性化约束
通过雅可比矩阵 $J_V$ 将顶点位置的变化线性化：
$$\Delta V \approx J_V \cdot \Delta q_a$$
从而 Laplacian 坐标的变化为：
$$\Delta L \approx L \cdot J_V \cdot \Delta q_a$$

在代码中体现为：
```python
# 构造 Laplacian 雅可比 (InteractionMeshRetargeter.py)
Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
J_L = Kron @ J_V
# 线性相等约束: J_L * dqa - lap_var == -lap0_vec
constraints += [cp.Constant(J_L[:, self.q_a_indices]) @ dqa - lap_var == -lap0_vec]
```

## 4. 总结

该方法的核心思想是通过保持关键点之间的**相对几何关系**（由 Laplacian 坐标表征）来实现动作的重定向。
1. **拓扑结构**：通过 `adj_list` 定义关键点（机器人关节点和物体点）之间的交互关系。
2. **局部特征保存**：Laplacian 坐标捕捉了每个点相对于其邻居的局部结构，最小化 $E_L$ 意味着在重定向过程中尽可能保留这些交互特征。
3. **优化求解**：利用 CVXPY 求解带约束的二次规划问题，在满足机器人物理约束（如关节限位、自碰撞）的同时，最小化几何变形能量。