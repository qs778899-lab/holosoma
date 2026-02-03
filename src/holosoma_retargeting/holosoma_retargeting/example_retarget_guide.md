#  Nokov格式的BVH文件的Retargeting重定向完整流程


## 1. 准备工作

首先，确保安装了必要的依赖项，并获取 LAFAN 数据处理工具。

```bash
cd src/holosoma_retargeting/holosoma_retargeting/data_utils/

# 克隆 LAFAN 官方仓库以获取处理脚本（如果尚未安装）
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
rm -rf ubisoft-laforge-animation-dataset
```

## 2. 提取全局坐标 (BVH -> NPY)

重定向流水线需要世界坐标系下的关节点位置。

cd holosoma/src/holosoma_retargeting/holosoma_retargeting

```bash
conda activate hsretargeting
python data_utils/extract_global_positions.py \
  --input_dir snooker \
  --output_dir snooker_npy \
  --data_format nokov
```

## 3. 执行批量重定向

转换完成后，使用并行重定向脚本处理所有序列。

```bash
cd ..

python examples/parallel_robot_retarget.py \
  --data-dir snooker_npy \
  --task-type robot_only \
  --data_format nokov \
  --save_dir snooker_results \
  --task-config.object-name ground \
  --task-config.ground-range -5 5 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --max-workers 12 

#增加左手腕的绝对旋转角度跟踪，不涉及position
python examples/parallel_robot_retarget.py \
  --data-dir snooker_npy \
  --task-type robot_only \
  --data_format nokov \
  --save_dir snooker_results \
  --task-config.object-name ground \
  --task-config.ground-range -5 5 \
  --retargeter.activate-snooker-tracking True \
  --retargeter.activate-snooker-laplacian False \
  --retargeter.activate-realtime-rotation-tracking False \
  --retargeter.activate-general-nominal-tracking False \
  --retargeter.snooker-frame-range 0 1680 \
  --max-workers 12

#进一步增加球杆约束
python examples/parallel_robot_retarget.py \
  --data-dir snooker_npy \
  --task-type robot_only \
  --data_format nokov \
  --save_dir snooker_results \
  --task-config.object-name ground \
  --task-config.ground-range -5 5 \
  --retargeter.activate-snooker-tracking True \
  --retargeter.activate-snooker-laplacian True \
  --retargeter.activate-realtime-rotation-tracking False \
  --retargeter.activate-general-nominal-tracking False \
  --retargeter.laplacian-frame-range 580 1300 \
  --retargeter.wrist-tracking-frame-range 0 1704 \
  --retargeter.snooker-frame-range 580 1300 \
  --max-workers 12

```


**参数说明：**
- python examples/parallel_robot_retarget.py: 调用并行重定向主脚本，利用多核 CPU 同时处理多个文件。
- --data-dir: 输入目录，脚本会遍历该目录下所有的 .npy 运动文件。
- --task-type robot_only: 任务模式。robot_only 表示仅重定向机器人动作，不涉及物体交互或复杂地形。
- --data_format : 输入数据格式。比如告知脚本以 LAFAN或者nokov 的骨骼结构和坐标规范来解析 .npy 文件。
- --save_dir: 结果保存目录。每个 BVH 序列处理完后，会在此生成对应的 .npz 文件。
- --task-config.object-name ground: 指定交互对象为地面。在 robot_only 模式下，这是保证机器人不穿模地面的关键配置。
- --task-config.ground-range -10 10: 地面判定范围。定义从 -10m 到 10m 的区域为有效接触平面。
- --retargeter.foot-sticking-tolerance 0.02: 足部贴地容差（单位：米）。当人体足部距离地面小于 2cm 时，算法会锁定机器人足部，防止产生“滑步”或“漂浮”感。数值越小要求越严苛，通常 0.01~0.03 之间效果较好。

## 4. 可视化结果

重定向完成后，你可以通过以下命令查看生成的机器人运动效果：

```bash

python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --qpos_npz snooker_results/snooker2_original.npz
  
python viser_player.py \
  --mjcf_path models/g1/scene_29dof_cue.xml \
  --qpos_npz snooker_results/snooker2_original.npz













```

---














## 5. 使用 Climbing 模式进行台球桌场景重定向

当需要让机器人与静态物体（如台球桌）进行交互时，可以使用 `climbing` 任务类型。

### 5.1 目录结构

台球桌场景的文件在 `demo_data/snooker/snooker_table/` 目录下：

```
demo_data/snooker/snooker_table/
│
│  ═══════════════ Retargeting 核心文件（MuJoCo 仿真） ═══════════════
│
├── g1_29dof_w_snooker_table.xml   # 完整场景 XML（机器人 + 球桌）
│   ├── <include file="box_assets.xml"/>
│   └── <include file="box_body.xml"/>
│
├── box_assets.xml                 # MuJoCo 资产定义
│   └── <mesh file="snooker_table.obj"/>
│
├── box_body.xml                   # MuJoCo 几何体定义（静态 geom）
│   └── 引用 box_assets.xml 中定义的 mesh
│
├── snooker_table.obj              # 3D mesh 文件（顶点 + 面片）
│   └── 被 MuJoCo 加载 + 被 load_object_data() 用于表面点采样
│
│  ═══════════════ 可视化文件（Viser） ═══════════════
│
├── snooker_table.urdf             # 球桌 URDF 文件
│   └── 引用 snooker_table.obj
│
│  ═══════════════ 工具和数据 ═══════════════
│
├── generate_snooker_mesh.py       # 生成 snooker_table.obj 的脚本
└── your_motion_data.npy           # 放置你的 Mocap 数据
```

**文件引用关系：**
```
g1_29dof_w_snooker_table.xml  ──include──►  box_assets.xml  ──file──►  snooker_table.obj
                              ──include──►  box_body.xml    ──mesh──►  (引用 box_assets.xml 中的 mesh)
```

> **注意**: 
> - Retargeting 时 MuJoCo 加载 `g1_29dof_w_snooker_table.xml`，它通过 include 依赖其他文件
> - 可视化时使用 `models/g1/scene_29dof_cue.xml`（G1 适配尺寸）或 `snooker_table.urdf`

### 5.2 球桌参数说明

球桌位置和尺寸（基于 `scene_29dof_cue.xml`）：
- **世界位置**: (0, -1.6, 0.48) - 在机器人前方 1.6m，桌面高度 0.48m
- **桌面尺寸**: 1.4m × 1.0m × 0.08m
- **桌腿**: 4 根圆柱，半径 0.04m，高度 0.8m

### 5.3 运行重定向命令

将你的 Mocap 数据（.npy 文件）放入 `snooker_table/` 目录后，运行：

```bash
cd holosoma/src/holosoma_retargeting/holosoma_retargeting

python examples/parallel_robot_retarget.py \
  --data-dir demo_data/snooker \
  --task-type climbing \
  --data_format nokov \
  --save_dir snooker_climbing_results \
  --task-config.object-name snooker_table \
  --task-config.object-dir demo_data/snooker/snooker_table \
  --task-config.surface-weight-threshold 0.0 \
  --task-config.surface-weight-high 20 \
  --task-config.surface-weight-low 1 \
  --retargeter.activate-snooker-tracking False \
  --retargeter.activate-snooker-laplacian False \
  --retargeter.activate-realtime-rotation-tracking False \
  --retargeter.activate-general-nominal-tracking False \
  --retargeter.activate-obj-non-penetration \
  --retargeter.snooker-frame-range 0 1300 \
  --max-workers 12
```

### 5.4 参数说明

| 参数 | 说明 |
|------|------|
| `--task-type climbing` | 使用静态物体交互模式 |
| `--task-config.object-name snooker_table` | 物体名称，对应 `{name}.obj` 和 `{name}.urdf` 文件 |
| `--task-config.object-dir` | 包含物体定义文件的目录路径 |
| `--task-config.surface-weight-threshold` | 见下方详解 |
| `--task-config.surface-weight-high` | 见下方详解 |
| `--task-config.surface-weight-low` | 见下方详解 |
| `--retargeter.activate-obj-non-penetration` | 启用物体穿透检测（flag 格式，不需要 True/False） |

#### 表面采样权重参数详解

这些参数控制从 mesh 文件采样交互点时的权重分布：

```python
# 采样逻辑（简化）
weight = surface_weight_high if point.z > threshold else surface_weight_low
```

**球桌 mesh 的局部坐标系**（以桌面中心为原点）：
```
z=+0.04  ──────────────  桌面顶部（交互表面）
z= 0.00  ──────────────  桌面中心  
z=-0.04  ──────────────  桌面底部
z=-0.84  ┴──────────────  桌腿底部
```

**推荐设置**：
- `threshold=0.0`：z > 0 的点（桌面顶部）获得高权重
- `high=20, low=1`：桌面顶部被采样的概率是桌腿的 **20 倍**


### 5.5 可视化结果

```bash

python viser_player.py \
  --mjcf_path demo_data/snooker/snooker_table/g1_29dof_w_snooker_table_scaled_0.74_0.74_0.74.xml \
  --qpos_npz snooker_climbing_results/snooker_original.npz
```

### 5.6 自定义球桌参数

如需修改球桌尺寸或位置，编辑以下文件：

1. **修改 mesh**: 编辑 `generate_snooker_mesh.py` 中的参数，重新运行生成新的 `.obj` 文件
2. **修改位置**: 同时更新以下文件中的位置：
   - `box_body.xml`: `pos="x y z"` 属性
   - `snooker_table.urdf`: `<origin xyz="x y z" .../>` 
   - `scene_29dof_cue.xml`（可选，用于可视化）

