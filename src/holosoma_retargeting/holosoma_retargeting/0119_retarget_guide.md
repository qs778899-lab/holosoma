# 0119 文件夹 BVH 数据重定向完整流程

本指南介绍了如何将 `0119` 文件夹中的 BVH 格式人体运动数据重定向到 G1 机器人。

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

重定向流水线需要世界坐标系下的关节点位置。使用以下命令将 `0119` 中的 `.bvh` 文件转换为 `.npy` 格式。

```bash
# 在 data_utils 目录下执行
python extract_global_positions.py \
  --input_dir ../0119 \
  --output_dir ../0119_npy
```

## 3. 执行批量重定向

转换完成后，使用并行重定向脚本处理所有序列。我们将结果保存到 `0119_results` 文件夹中。

```bash
# 返回到 holosoma_retargeting 根目录
cd ..

# 执行批量重定向
python examples/parallel_robot_retarget.py \
  --data-dir 0119_npy \
  --task-type robot_only \
  --data_format lafan \
  --save_dir 0119_results \
  --task-config.object-name ground \
  --task-config.ground-range -10 10 \
  --retargeter.foot-sticking-tolerance 0.02
```

**参数说明：**
- `--data-dir`: 输入的 `.npy` 文件目录。
- `--data_format lafan`: 指定数据格式为 LAFAN (对应 `.npy` 全局坐标)。
- `--retargeter.foot-sticking-tolerance 0.02`: 足部贴地容差，可根据实际效果微调。

## 4. 可视化结果

重定向完成后，你可以通过以下命令查看生成的机器人运动效果：

```bash
# 替换为你想要查看的文件名
python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --qpos_npz 0119_results/SIK337_zou_20251217_1648_original.npz
```

## 5. (可选) 转换为 RL 训练格式

如果你需要将这些数据用于强化学习训练（如 Whole-Body Tracking），请执行：

```bash
python data_conversion/convert_data_format_mj.py \
  --input_file 0119_results/SIK337_zou_20251217_1648_original.npz \
  --output_fps 50 \
  --output_name 0119_converted/SIK337_zou_mj_fps50.npz \
  --data_format lafan \
  --object_name "ground" \
  --once
```

