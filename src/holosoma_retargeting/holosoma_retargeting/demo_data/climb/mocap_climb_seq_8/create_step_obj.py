def generate_staircase_obj(num_steps, start_x=0.14, end_x=1.73, step_width=0.5, step_height=0.44):
    vertices = []
    faces = []
    
    current_y = 2.0 # 机器人面向台阶最最近的坐标
    current_z = 0.0 # 地面高度
    
    for i in range(num_steps):
        # 每一级台阶的基础索引偏移
        offset = i * 8
        
        # 按照你的规律定义 8 个顶点
        y_min, y_max = current_y, current_y + step_width
        z_min, z_max = current_z, current_z + step_height
        
        # 顶点坐标 (v)
        v = [
            (start_x, y_min, z_min), # 1
            (start_x, y_min, z_max), # 2
            (start_x, y_max, z_min), # 3
            (start_x, y_max, z_max), # 4
            (end_x,   y_min, z_min), # 5
            (end_x,   y_min, z_max), # 6
            (end_x,   y_max, z_min), # 7
            (end_x,   y_max, z_max)  # 8
        ]
        vertices.extend(v)
        
        # 按照你的规律定义 12 个三角面 (f)
        # 将原始索引加上偏移量
        f_template = [
            (1, 2, 4), (1, 4, 3), # 左侧面
            (5, 7, 8), (5, 8, 6), # 右侧面
            (1, 5, 6), (1, 6, 2), # 前侧面
            (3, 4, 8), (3, 8, 7), # 后侧面
            (1, 3, 7), (1, 7, 5), # 底面
            (2, 6, 8), (2, 8, 4)  # 顶面
        ]
        
        for face in f_template:
            faces.append(tuple(idx + offset for idx in face))
            
        # 楼梯逻辑：下一级台阶在 Y 方向和 Z 方向递增
        current_y += step_width
        current_z += step_height

    # 生成 OBJ 内容
    output = ["# Staircase OBJ generated with custom logic"]
    for v in vertices:
        output.append(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}")
    
    for f in faces:
        output.append(f"f {f[0]} {f[1]} {f[2]}")
        
    return "\n".join(output)

# 生成 3 级台阶示例
print(generate_staircase_obj(3))
