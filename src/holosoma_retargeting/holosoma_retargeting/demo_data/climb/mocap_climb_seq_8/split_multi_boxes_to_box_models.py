#!/usr/bin/env python3
"""
从 multi_boxes.obj 拆出 box_models/box1.obj, box2.obj, ... 使与合并 mesh 一致。

约定：multi_boxes.obj 中每 8 个顶点为一级台阶，每 12 个三角面为一级（顶点索引连续）。
用法：
  python examples/split_multi_boxes_to_box_models.py demo_data/climb/mocap_sample
或指定 obj 与输出目录：
  python examples/split_multi_boxes_to_box_models.py --obj path/to/multi_boxes.obj --out path/to/box_models
"""

from __future__ import annotations

import argparse
from pathlib import Path

VERTICES_PER_BOX = 8
FACES_PER_BOX = 12


def parse_obj(obj_path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Parse OBJ: return list of (x,y,z), list of (a,b,c) 1-based face indices."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(obj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f" and len(parts) >= 4:
                # f v1 v2 v3 (1-based); allow v1/vt1/vn1 format by taking first number
                def idx(s: str) -> int:
                    return int(s.split("/")[0])
                faces.append((idx(parts[1]), idx(parts[2]), idx(parts[3])))
    return vertices, faces


def split_boxes(vertices: list, faces: list) -> list[tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]]:
    """Split into per-box vertex and face lists. Faces are remapped to 1-based indices 1..8 per box."""
    n_vertices = len(vertices)
    n_boxes = n_vertices // VERTICES_PER_BOX
    if n_boxes * VERTICES_PER_BOX != n_vertices:
        raise ValueError(f"Vertex count {n_vertices} is not a multiple of {VERTICES_PER_BOX}")
    boxes: list[tuple[list, list]] = []
    for k in range(n_boxes):
        start = k * VERTICES_PER_BOX
        end = start + VERTICES_PER_BOX
        box_verts = vertices[start:end]
        # Faces that reference only vertices in [start+1, end] (1-based)
        box_faces: list[tuple[int, int, int]] = []
        for (a, b, c) in faces:
            if all(start + 1 <= i <= end for i in (a, b, c)):
                box_faces.append((a - start, b - start, c - start))
        if len(box_faces) != FACES_PER_BOX:
            raise ValueError(
                f"Box {k + 1}: expected {FACES_PER_BOX} faces, got {len(box_faces)}. "
                "multi_boxes.obj may not follow 8 vertices + 12 faces per box."
            )
        boxes.append((box_verts, box_faces))
    return boxes


def write_obj(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]], comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        for v in vertices:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split multi_boxes.obj into box_models/box1.obj, box2.obj, ...")
    parser.add_argument(
        "task_dir",
        nargs="?",
        default="demo_data/climb/mocap_sample",
        help="Task directory containing multi_boxes.obj; box_models/ will be created/updated inside it.",
    )
    parser.add_argument("--obj", type=Path, help="Path to multi_boxes.obj (default: <task_dir>/multi_boxes.obj)")
    parser.add_argument("--out", type=Path, help="Output directory for box N.obj (default: <task_dir>/box_models)")
    args = parser.parse_args()
    task_dir = Path(args.task_dir)
    obj_path = args.obj or (task_dir / "multi_boxes.obj")
    out_dir = args.out or (task_dir / "box_models")
    if not obj_path.exists():
        raise FileNotFoundError(f"multi_boxes.obj not found: {obj_path}")
    vertices, faces = parse_obj(obj_path)
    boxes = split_boxes(vertices, faces)
    for k, (box_verts, box_faces) in enumerate(boxes):
        out_path = out_dir / f"box{k + 1}.obj"
        write_obj(out_path, box_verts, box_faces, comment=f"box{k + 1} from {obj_path.name}")
        print(f"Wrote {out_path} ({len(box_verts)} vertices, {len(box_faces)} faces)")
    print(f"Done: {len(boxes)} boxes -> {out_dir}")


if __name__ == "__main__":
    main()
