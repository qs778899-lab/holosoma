#!/usr/bin/env python3
"""
Generated files:
- snooker_table.obj    # Complete table mesh (table top + 4 legs)

The table parameters are SCALED for 1.78m human body (will be auto-scaled to G1 1.32m by retargeting).
Scale factor: 1.78 / 1.32 = 1.348

Original values (for G1 robot 1.32m):
- Table position in world: (0, -1.9, 0.48)
- Table size: 1.4m x 1.0m x 0.08m
- Legs at corners: (±0.6, ±0.4, -0.4) relative to table center

Scaled values (for human 1.78m, used in this file):
- Table position in world: (0, -1.9 * SCALE, 0.65)
- Table size: 1.89m x 1.35m x 0.11m
- Legs at corners: (±0.81, ±0.54, -0.54) relative to table center

"""

# cd /home/s114/Retarget_Work/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/snooker/snooker_table && python generate_snooker_mesh.py


import numpy as np


def create_box_obj(width, depth, height, filename, center_offset=(0, 0, 0)):
    """
    Create a simple box mesh in OBJ format.
    
    Args:
        width: X dimension
        depth: Y dimension  
        height: Z dimension
        filename: Output .obj file path
        center_offset: (x, y, z) offset for the box center
    """
    hw, hd, hh = width / 2, depth / 2, height / 2
    ox, oy, oz = center_offset
    
    # 8 vertices of a box
    vertices = [
        (-hw + ox, -hd + oy, -hh + oz),  # 0: bottom-back-left
        ( hw + ox, -hd + oy, -hh + oz),  # 1: bottom-back-right
        ( hw + ox,  hd + oy, -hh + oz),  # 2: bottom-front-right
        (-hw + ox,  hd + oy, -hh + oz),  # 3: bottom-front-left
        (-hw + ox, -hd + oy,  hh + oz),  # 4: top-back-left
        ( hw + ox, -hd + oy,  hh + oz),  # 5: top-back-right
        ( hw + ox,  hd + oy,  hh + oz),  # 6: top-front-right
        (-hw + ox,  hd + oy,  hh + oz),  # 7: top-front-left
    ]
    
    # 6 faces (each face is 2 triangles, defined by vertex indices)
    # OBJ uses 1-based indexing
    faces = [
        # Bottom face (z = -hh)
        (1, 2, 3), (1, 3, 4),
        # Top face (z = +hh)
        (5, 7, 6), (5, 8, 7),
        # Front face (y = +hd)
        (4, 3, 7), (3, 6, 7),
        # Back face (y = -hd)
        (1, 5, 2), (2, 5, 6),
        # Left face (x = -hw)
        (1, 4, 8), (1, 8, 5),
        # Right face (x = +hw)
        (2, 6, 3), (3, 6, 7),
    ]
    
    # Correct faces for proper winding
    faces = [
        # Bottom face (z = -hh) - looking from below
        (1, 3, 2), (1, 4, 3),
        # Top face (z = +hh) - looking from above
        (5, 6, 7), (5, 7, 8),
        # Front face (y = +hd)
        (4, 7, 3), (3, 7, 6),
        # Back face (y = -hd)
        (1, 2, 5), (2, 6, 5),
        # Left face (x = -hw)
        (1, 5, 4), (4, 5, 8),
        # Right face (x = +hw)
        (2, 3, 6), (3, 7, 6),
    ]
    
    with open(filename, 'w') as f:
        f.write(f"# Snooker table mesh\n")
        f.write(f"# Dimensions: {width} x {depth} x {height}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Created {filename}")
    return vertices


def create_cylinder_obj(radius, height, segments, filename, center_offset=(0, 0, 0)):
    """
    Create a cylinder mesh in OBJ format.
    
    Args:
        radius: Cylinder radius
        height: Cylinder height (along Z axis)
        segments: Number of segments around the circumference
        filename: Output .obj file path
        center_offset: (x, y, z) offset for the cylinder center
    """
    ox, oy, oz = center_offset
    hh = height / 2
    
    vertices = []
    
    # Top center
    vertices.append((ox, oy, oz + hh))
    # Bottom center
    vertices.append((ox, oy, oz - hh))
    
    # Generate vertices around the circumference
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle) + ox
        y = radius * np.sin(angle) + oy
        vertices.append((x, y, oz + hh))  # Top ring
        vertices.append((x, y, oz - hh))  # Bottom ring
    
    faces = []
    
    # Top and bottom caps
    for i in range(segments):
        next_i = (i + 1) % segments
        # Top cap (vertex indices: 0 is top center, 2+2*i is top ring)
        top_v1 = 3 + 2 * i  # 1-based index
        top_v2 = 3 + 2 * next_i
        faces.append((1, top_v2, top_v1))  # Top center is 1
        
        # Bottom cap
        bot_v1 = 4 + 2 * i  # 1-based index
        bot_v2 = 4 + 2 * next_i
        faces.append((2, bot_v1, bot_v2))  # Bottom center is 2
        
        # Side faces (2 triangles per segment)
        faces.append((top_v1, top_v2, bot_v1))
        faces.append((bot_v1, top_v2, bot_v2))
    
    with open(filename, 'w') as f:
        f.write(f"# Cylinder mesh\n")
        f.write(f"# Radius: {radius}, Height: {height}, Segments: {segments}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Created {filename}")


def create_snooker_table_mesh():
    """
    Create the complete snooker table mesh.
    
    SCALED for 1.78m human body (scale factor = 1.78/1.32 = 1.348):
    - Table top: 1.89m x 1.35m x 0.11m (original: 1.4m x 1.0m x 0.08m)
    - Legs: cylinder radius=0.054, height=0.596m
    
    WORLD OFFSET (to match URDF visual origin):
    - Table center: (0, -1.9 * SCALE, 0.65)
    """
    
    # Scale factor for human body (1.78m) vs G1 robot (1.32m)
    SCALE = 1.78 / 1.32  # ≈ 1.348
    
    # World offsets (Human scale)
    WORLD_X = 0.0
    WORLD_Y = -1.9 * SCALE
    WORLD_Z = 0.836
    
    # Table dimensions (in object local coordinates) - SCALED
    table_width = 1.4 * SCALE   # X direction: ~1.89m
    table_depth = 1.0 * SCALE   # Y direction: ~1.35m
    table_height = 0.08 * SCALE  # Z direction (thickness): ~0.11m
    
    # Leg dimensions - SCALED
    leg_radius = 0.04 * SCALE   # ~0.054m
    # Adjusted leg height to touch ground (Z=0) while keeping table top at Z=0.89 (human scale)
    # Table top center is at 0.836, thickness is 0.108. Bottom surface is at 0.836 - 0.054 = 0.782.
    leg_height = 0.782  # Full height
    leg_segments = 16
    
    # Leg positions (relative to table center) - SCALED
    # Original: (±0.6, ±0.4, -0.33)
    leg_offset_x = 0.6 * SCALE  # ~0.81
    leg_offset_y = 0.4 * SCALE  # ~0.54
    # Leg center Z relative to table top center (0.836)
    # Leg center is at 0.391. Offset = 0.391 - 0.836 = -0.445
    leg_offset_z = -0.445
    
    leg_positions = [
        (leg_offset_x, leg_offset_y, leg_offset_z),    # Front-right
        (leg_offset_x, -leg_offset_y, leg_offset_z),   # Back-right
        (-leg_offset_x, leg_offset_y, leg_offset_z),   # Front-left
        (-leg_offset_x, -leg_offset_y, leg_offset_z),  # Back-left
    ]
    
    # Create combined mesh
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # === Table Top ===
    # Center the table top at (WORLD_X, WORLD_Y, WORLD_Z)
    hw, hd, hh = table_width / 2, table_depth / 2, table_height / 2
    table_vertices = [
        (-hw + WORLD_X, -hd + WORLD_Y, -hh + WORLD_Z),
        ( hw + WORLD_X, -hd + WORLD_Y, -hh + WORLD_Z),
        ( hw + WORLD_X,  hd + WORLD_Y, -hh + WORLD_Z),
        (-hw + WORLD_X,  hd + WORLD_Y, -hh + WORLD_Z),
        (-hw + WORLD_X, -hd + WORLD_Y,  hh + WORLD_Z),
        ( hw + WORLD_X, -hd + WORLD_Y,  hh + WORLD_Z),
        ( hw + WORLD_X,  hd + WORLD_Y,  hh + WORLD_Z),
        (-hw + WORLD_X,  hd + WORLD_Y,  hh + WORLD_Z),
    ]
    
    table_faces = [
        # Bottom
        (1, 3, 2), (1, 4, 3),
        # Top
        (5, 6, 7), (5, 7, 8),
        # Front (y+)
        (4, 8, 3), (3, 8, 7),
        # Back (y-)
        (1, 2, 5), (2, 6, 5),
        # Left (x-)
        (1, 5, 4), (4, 5, 8),
        # Right (x+)
        (2, 3, 6), (3, 7, 6),
    ]
    
    all_vertices.extend(table_vertices)
    all_faces.extend(table_faces)
    vertex_offset = len(table_vertices)
    
    # === Legs ===
    for leg_pos in leg_positions:
        # leg_pos is relative to table center (WORLD_X, WORLD_Y, WORLD_Z)
        ox, oy, oz = leg_pos[0] + WORLD_X, leg_pos[1] + WORLD_Y, leg_pos[2] + WORLD_Z
        leg_hh = leg_height / 2
        
        # Leg vertices
        leg_verts = []
        # Top center
        leg_verts.append((ox, oy, oz + leg_hh))
        # Bottom center  
        leg_verts.append((ox, oy, oz - leg_hh))
        
        # Generate ring vertices
        for i in range(leg_segments):
            angle = 2 * np.pi * i / leg_segments
            x = leg_radius * np.cos(angle) + ox
            y = leg_radius * np.sin(angle) + oy
            leg_verts.append((x, y, oz + leg_hh))  # Top ring
            leg_verts.append((x, y, oz - leg_hh))  # Bottom ring
        
        # Leg faces (with offset)
        leg_faces = []
        for i in range(leg_segments):
            next_i = (i + 1) % leg_segments
            
            top_v1 = vertex_offset + 3 + 2 * i
            top_v2 = vertex_offset + 3 + 2 * next_i
            bot_v1 = vertex_offset + 4 + 2 * i
            bot_v2 = vertex_offset + 4 + 2 * next_i
            top_center = vertex_offset + 1
            bot_center = vertex_offset + 2
            
            # Top cap
            leg_faces.append((top_center, top_v2, top_v1))
            # Bottom cap
            leg_faces.append((bot_center, bot_v1, bot_v2))
            # Sides
            leg_faces.append((top_v1, top_v2, bot_v1))
            # Bottom sides
            leg_faces.append((bot_v1, top_v2, bot_v2))
        
        all_vertices.extend(leg_verts)
        all_faces.extend(leg_faces)
        vertex_offset += len(leg_verts)

    
    # Write combined mesh
    filename = "snooker_table.obj"
    with open(filename, 'w') as f:
        f.write("# Snooker Table Mesh\n")
        f.write("# Table top: 1.4m x 1.0m x 0.08m, centered at origin\n")
        f.write("# 4 legs: radius=0.04m, height=0.8m\n")
        f.write(f"# Total vertices: {len(all_vertices)}\n")
        f.write(f"# Total faces: {len(all_faces)}\n\n")
        
        for v in all_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        for face in all_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Created {filename}")
    print(f"  Vertices: {len(all_vertices)}")
    print(f"  Faces: {len(all_faces)}")


if __name__ == "__main__":
    create_snooker_table_mesh()
    print("\nMesh file created successfully!")
    print("Note: Surface sampling uses surface_weight_threshold parameter to weight table top higher.")

