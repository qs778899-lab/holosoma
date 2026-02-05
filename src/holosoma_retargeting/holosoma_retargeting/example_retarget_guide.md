#  Nokov格式的BVH文件的Retargeting重定向完整流程


## 1. 准备工作

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

## 3. 执行重定向

```bash

python examples/parallel_robot_retarget.py \
  --data-dir snooker_npy \
  --task-type robot_only \
  --data_format nokov \
  --save_dir snooker_results \
  --task-config.object-name ground \
  --task-config.ground-range -10 10 \
  --retargeter.activate-snooker-tracking True \
  --retargeter.activate-snooker-laplacian False \
  --retargeter.activate-realtime-rotation-tracking False \
  --retargeter.activate-general-nominal-tracking False \
  --retargeter.snooker-frame-range 0 1680 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --max-workers 12


python examples/robot_retarget.py \
  --data-path snooker_npy \
  --task-name snooker2 \
  --task-type robot_only \
  --data-format nokov \
  --save-dir snooker_results \
  --task-config.object-name ground \
  --task-config.ground-range -10 10 \
  --retargeter.activate-snooker-tracking False \
  --retargeter.activate-snooker-laplacian True \
  --retargeter.activate-realtime-rotation-tracking False \
  --retargeter.activate-general-nominal-tracking False \
  --retargeter.laplacian-frame-range 580 1300 \
  --retargeter.wrist-tracking-frame-range 0 1704 \
  --retargeter.snooker-frame-range 580 1300 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --retargeter.visualize \
  --retargeter.debug \
  --retargeter.visualization-interp-mult 1 \
  --retargeter.smooth-weight 10.0 

```

**参数说明：**
- --task-type robot_only/object_interaction/climbing: 任务模式
- --task-config.object-name ground/largebox/multi_boxes: 交互对象
- --task-config.ground-range -10 10: 定义了虚拟地面网格的物理范围，用于防止机器人脚部穿透地面
- --retargeter.foot-sticking-tolerance 0.02: 足部贴地容差（单位：米）。当人体足部距离地面小于 2cm 时，算法会锁定机器人足部，防止产生“滑步”或“漂浮”感。
可视化：蓝色点是人体关键点，绿色点是机器人实际点


## 4. 可视化结果

```bash

python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --qpos_npz snooker_results/snooker2_original.npz
  
python viser_player.py \
  --mjcf_path models/g1/scene_29dof_cue.xml \
  --qpos_npz snooker_results/snooker2.npz

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
├── g1_29dof_w_snooker_table.xml   # 完整场景架构的XML，但球桌的具体定义在include文件中
│   ├── <include file="box_assets.xml"/> 
│   └── <include file="box_body.xml"/>   
│
├── box_assets.xml                 # MuJoCo 资产定义
│   └── <mesh file="snooker_table.obj"/>
│
├── box_body.xml                   # MuJoCo 几何体定义如位置信息（静态 geom）
│   └── 引用 box_assets.xml 中定义的 mesh
│    
├── snooker_table.obj              # 3D mesh 文件（顶点 + 面片）
│   └── 被 MuJoCo 加载 + 被 load_object_data() 用于表面点采样
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
实际 retarget 会用到的模型文件（缩放后）：g1_29dof_w_snooker_table_scaled_0.74_0.74_0.74.xml, box_assets_scaled_0.74_0.74_0.74.xml, box_body_scaled_0.74_0.74_0.74.xml, snooker_table.obj（不生成新 obj文件，只在加载时缩放。因为缩放是在 box_assets_scaled_*.xml 的 mesh scale="..." 上完成的；另外交互点采样也会用 smpl_scale 直接缩放点云）。




### 5.2 球桌参数说明

球桌位置和尺寸（基于 `scene_29dof_cue.xml`）：
- **世界位置**: (0, -1.6, 0.48) - 在机器人前方 1.6m，桌面高度 0.48m
- **桌面尺寸**: 1.4m × 1.0m × 0.08m
- **桌腿**: 4 根圆柱，半径 0.04m，高度 0.8m

### 5.3 运行重定向命令

将你的 Mocap 数据（.npy 文件）放入 `snooker_table/` 目录后，运行：

```bash
cd holosoma/src/holosoma_retargeting/holosoma_retargeting

python examples/robot_retarget.py \
  --data-path snooker_npy \
  --task-name snooker3 \
  --task-type climbing \
  --data-format nokov \
  --save-dir snooker_results \
  --task-config.object-name snooker_table \
  --task-config.object-dir demo_data/snooker/snooker_table \
  --task-config.ground-range -10 10 \
  --task-config.surface-weight-threshold 0.0 \
  --task-config.surface-weight-high 20 \
  --task-config.surface-weight-low 1 \
  --retargeter.activate-snooker-tracking False \
  --retargeter.activate-snooker-laplacian True \
  --retargeter.activate-realtime-rotation-tracking False \
  --retargeter.activate-general-nominal-tracking False \
  --retargeter.activate-obj-non-penetration \
  --retargeter.laplacian-frame-range 580 1300 \
  --retargeter.wrist-tracking-frame-range 0 1704 \
  --retargeter.snooker-frame-range 580 1300 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --retargeter.visualize \
  --retargeter.debug \
  --retargeter.visualization-interp-mult 1 \
  --retargeter.smooth-weight 10.0 

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

