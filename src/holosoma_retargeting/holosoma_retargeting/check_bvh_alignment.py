import os
import sys
import numpy as np
from pathlib import Path

#python check_bvh_alignment.py


#python check_bvh_alignment.py snooker/snooker2.bvh
#python check_bvh_alignment.py climb/climb59.bvh

# Add the current directory and its parent to sys.path to import local modules
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from data_utils.lafan1 import extract, utils
    from config_types.data_type import LAFAN_DEMO_JOINTS
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    # Fallback if the above fails depending on how the script is run
    sys.path.insert(0, str(current_dir.parent))
    from holosoma_retargeting.data_utils.lafan1 import extract, utils
    from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS

def check_bvh(bvh_path):
    print(f"\n{'='*20} Checking BVH: {os.path.basename(bvh_path)} {'='*20}")
    
    # 1. Read BVH
    try:
        anim = extract.read_bvh(bvh_path)
    except Exception as e:
        print(f"Error reading BVH: {e}")
        return

    # 2. Check Joints
    print(f"\n[1] Joint Analysis:")
    print(f"Total joints in BVH: {len(anim.bones)}")
    print(f"Expected joints (LAFAN): {len(LAFAN_DEMO_JOINTS)}")
    
    bvh_joints = anim.bones
    print("\n--- Full Joint List & Hierarchy (Index: Name [Parent]) ---")
    for idx, name in enumerate(bvh_joints):
        parent_idx = anim.parents[idx]
        parent_name = bvh_joints[parent_idx] if parent_idx != -1 else "None (Root)"
        print(f"{idx:2d}: {name:<20} [Parent: {parent_name}]")
    print("----------------------------------------------------------\n")
    
    missing_joints = [j for j in LAFAN_DEMO_JOINTS if j not in bvh_joints]
    extra_joints = [j for j in bvh_joints if j not in LAFAN_DEMO_JOINTS]
    
    if not missing_joints:
        print("✅ All expected LAFAN joints found.")
    else:
        print(f"❌ Missing joints: {missing_joints}")
        
    print(f"ℹ️ Extra joints found: {len(extra_joints)} (including {extra_joints[:5]}...)")

    # 3. Check Units & Axis (using global positions)
    # Compute global positions using Forward Kinematics
    _, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    
    first_frame = global_positions[0]
    mins = np.min(first_frame, axis=0)
    maxs = np.max(first_frame, axis=0)
    height = maxs[1] - mins[1] if maxs[1] - mins[1] > maxs[2] - mins[2] else maxs[2] - mins[2]
    
    print(f"\n[2] Units & Dimensions (First Frame):")
    print(f"Min coords: {mins}")
    print(f"Max coords: {maxs}")
    print(f"Approximate height: {height:.2f}")
    
    if height > 50:
        print("💡 Suggestion: Units seem to be CENTIMETERS (Height > 50).")
    elif height > 0.5:
        print("💡 Suggestion: Units seem to be METERS.")
    else:
        print("❓ Suggestion: Units are unclear or scale is very small.")

    # 4. Check Root Position
    root_pos = anim.pos[0, 0]
    print(f"\n[3] Root Position (First Frame):")
    print(f"Hips position: {root_pos}")
    
    # 5. Check All Joint Definitions (Raw Hierarchy)
    print(f"\n[4] Complete BVH Hierarchy Definitions:")
    with open(bvh_path, 'r') as f:
        hierarchy_lines = []
        in_hierarchy = False
        for line in f:
            if "HIERARCHY" in line:
                in_hierarchy = True
                continue
            if "MOTION" in line:
                break
            if in_hierarchy:
                print(f"  {line.rstrip()}")
    
    print("\n" + "="*60)
    
    # 6. Check Specific Joint Global Positions (First Frame)
    print(f"\n[5] Specific Joint Global Positions (First Frame):")
    target_debug_joints = ["LeftFoot", "RightFoot"]
    for name in target_debug_joints:
        if name in anim.bones:
            idx = anim.bones.index(name)
            pos = global_positions[0, idx]
            print(f"Index {idx:2d} | Name: {name:<20} | Global Pos (m): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        else:
            print(f"❌ Joint {name} not found in BVH.")

    # 6. Check Axis Convention
    # In BVH, Y is typically up. Let's see which axis has the largest spread for "height"
    y_spread = maxs[1] - mins[1]
    z_spread = maxs[2] - mins[2]
    if y_spread > z_spread:
        print("💡 Axis Convention: Likely Y-UP (standard BVH).")
    else:
        print("💡 Axis Convention: Likely Z-UP.")

    # 7. Check Motion Data (First Frame)
    print(f"\n[6] Motion Data Analysis (First Frame):")
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
        
    in_motion = False
    motion_start_idx = -1
    for i, line in enumerate(lines):
        if "MOTION" in line:
            in_motion = True
            continue
        if in_motion:
            if "Frames:" in line:
                print(f"  {line.strip()}")
            elif "Frame Time:" in line:
                print(f"  {line.strip()}")
            elif line.strip() and not any(c.isalpha() for c in line.strip().split()[0]):
                motion_start_idx = i
                break
    
    if motion_start_idx != -1:
        first_frame_raw = lines[motion_start_idx].strip().split()
        print(f"  Total Channels in Motion Line: {len(first_frame_raw)}")
        
        # --- NEW: Build Channel Mapping ---
        print(f"\n  --- Motion Channel Mapping (First 20 Channels) ---")
        channel_map = []
        current_joint = None
        
        # Simple parser to find joint names and their channel counts
        with open(bvh_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("ROOT") or line.startswith("JOINT"):
                    current_joint = line.split()[1]
                elif line.startswith("CHANNELS"):
                    parts = line.split()
                    count = int(parts[1])
                    names = parts[2:]
                    for name in names:
                        channel_map.append(f"{current_joint}.{name}")
        
        for idx in range(min(20, len(channel_map))):
            val = first_frame_raw[idx] if idx < len(first_frame_raw) else "N/A"
            print(f"  Index {idx:2d} | {channel_map[idx]:<30} | Value: {val}")
        
        print(f"  ... (Mapping continues for all 180 channels) ...")
        
        print(f"\n  First Frame Raw Data (Full):")
        # 打印完整数据，每 10 个数值换一行以便阅读
        for i in range(0, len(first_frame_raw), 10):
            chunk = first_frame_raw[i : i + 10]
            print(f"    {' '.join(chunk)}")
        
        # Breakdown of the first few channels (typically Hips)
        print(f"\n  --- Data Format Breakdown (Initial Root Channels) ---")
        # In BVH, the first joint (Hips) usually has 6 channels: Xpos Ypos Zpos Yrot Xrot Zrot
        channel_names = ["Xpos", "Ypos", "Zpos", "Yrot", "Xrot", "Zrot"]
        for idx, name in enumerate(channel_names):
            if idx < len(first_frame_raw):
                print(f"  Channel {idx:2d}: {name:<10} = {first_frame_raw[idx]}")
    else:
        print("❌ Could not find motion data in BVH.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample_bvh = Path(sys.argv[1])
    else:
        sample_bvh = current_dir / "0119" / "SIK337_zou_20251217_1648.bvh"
        if not sample_bvh.exists():
            # Try relative to workspace
            sample_bvh = Path("src/holosoma_retargeting/holosoma_retargeting/0119/SIK337_zou_20251217_1648.bvh")
        
    if sample_bvh.exists():
        check_bvh(str(sample_bvh))
    else:
        print(f"Could not find sample BVH at {sample_bvh}")

