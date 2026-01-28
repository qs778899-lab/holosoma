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
# 执行批量重定向 (针对 Nokov/Snooker 数据)
python examples/parallel_robot_retarget.py \
  --data-dir snooker_npy \
  --task-type robot_only \
  --data_format nokov \
  --save_dir snooker_results \
  --task-config.object-name ground \
  --task-config.ground-range -5 5 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --max-workers 12 \
  --retargeter.snooker-frame-range 580  1300

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

python viser_player.py \
  --mjcf_path models/g1/scene_29dof_cue.xml \
  --qpos_npz snooker_results0126/snooker2_original.npz

```

