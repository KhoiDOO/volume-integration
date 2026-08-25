import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import sys

import conquer3d
from data import OptimizedRedwood
from conquer3d.data_structure.grid import get_active_voxel_ids_from_depth, build_sparse_grid_from_active_voxels
from conquer3d.ops import single_view_volume_integral, marching_cubes

DATASET_MAPPING = {
    "redwood": {
        "dataset_class": OptimizedRedwood,
        "scenes": ["apartment", "bedroom", "boardroom", "lobby", "office"]
    }
}

def calculate_bounds_from_poses(dataset, width: int = 640, height: int = 480, depth_max: float = 3.0, pad: float = 0.1):
    """
    Calculates the exact visible bounding box using the camera frustum at depth_max.
    Returns grid_min and grid_max in world coordinates.
    """
    import numpy as np
    
    # Get intrinsics (assuming Redwood dataset format)
    # We load one sample just to get the intrinsics without hardcoding them here
    intrinsics = dataset[0]["intrinsics"].numpy()
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    d = depth_max
    
    # Define the 5 corners of the camera frustum in local camera space
    frustum_cam = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],  # Camera origin
        [(0 - cx) * d / fx, (0 - cy) * d / fy, d, 1.0],  # Top-Left
        [(width - 1 - cx) * d / fx, (0 - cy) * d / fy, d, 1.0],  # Top-Right
        [(0 - cx) * d / fx, (height - 1 - cy) * d / fy, d, 1.0],  # Bottom-Left
        [(width - 1 - cx) * d / fx, (height - 1 - cy) * d / fy, d, 1.0],  # Bottom-Right
    ], dtype=torch.float32) # Shape: (5, 4)
    
    # Stack all Camera-to-World poses into a single batch tensor
    poses = torch.tensor(np.array(dataset.poses), dtype=torch.float32) # Shape: (N, 4, 4)
    
    # Batch multiply: (N, 4, 4) @ (4, 5) -> (N, 4, 5) -> transpose to (N, 5, 4)
    frustum_world = torch.matmul(poses, frustum_cam.T).transpose(1, 2)
    
    # Extract just the X, Y, Z coordinates (drop the homogeneous 1.0)
    frustum_world = frustum_world[..., :3] # Shape: (N, 5, 3)
    
    # Find the global min and max across all N poses and all 5 frustum points
    grid_min = torch.min(frustum_world.reshape(-1, 3), dim=0)[0]
    grid_max = torch.max(frustum_world.reshape(-1, 3), dim=0)[0]
    
    return grid_min - pad, grid_max + pad

@torch.inference_mode()
def volume_integration_pc2mesh(
    dataset_name: str,
    scene_name: str,
    data_dir: str, 
    voxel_size: float = 0.02, 
    trunc_margin: float = 0.04,
    mode: int = 1,
    use_color: bool = True
):
    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Available datasets: {list(DATASET_MAPPING.keys())}")
        
    if scene_name not in DATASET_MAPPING[dataset_name]["scenes"]:
        raise ValueError(f"Scene '{scene_name}' is not recognized for dataset '{dataset_name}'. Available scenes: {DATASET_MAPPING[dataset_name]['scenes']}")

    dataset_cls = DATASET_MAPPING[dataset_name]["dataset_class"]

    print(f"Loading {dataset_name}/{scene_name} from {data_dir}...")
    dataset = dataset_cls(data_dir, load_color=use_color)
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your data_dir path.")
    
    print("Calculating scene bounding box...")
    grid_min, grid_max = calculate_bounds_from_poses(dataset, pad=0.1)
    print(f"Bounds: Min {grid_min.tolist()} -> Max {grid_max.tolist()}")
    
    # --- ADD THIS TEMPORARY CROP ---
    center = (grid_max + grid_min) / 2.0
    grid_min = center - 1.0  # 1 meter left/down/back
    grid_max = center + 1.0  # 1 meter right/up/forward
    # -------------------------------
    
    # Calculate grid resolution based on bounding box
    res = torch.ceil((grid_max - grid_min) / voxel_size).int().tolist()
    print(f"Grid Resolution: {res}")
    
    # Create DataLoader for background image loading
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    
    print("Finding active voxels from all views...")
    active_mask = torch.zeros(res[0] * res[1] * res[2], dtype=torch.bool, device='cuda')
    
    for batch in tqdm(loader, total=len(dataset)):
        depth = batch["depth"][0].cuda(non_blocking=True).contiguous()
        c2w = batch["c2w"][0].cuda(non_blocking=True).contiguous()
        intrinsics = batch["intrinsics"][0].cuda(non_blocking=True).contiguous()
        intrinsics_inv = torch.linalg.inv(intrinsics).contiguous()
        
        ids = get_active_voxel_ids_from_depth(
            depth_image=depth, 
            c2w=c2w, 
            intrinsics_inv=intrinsics_inv, 
            grid_min=grid_min.tolist(), 
            grid_max=grid_max.tolist(), 
            res=res, 
            activate_neighbor=True, 
            trunc_margin=trunc_margin
        )
        active_mask[ids] = True
        
        # Free the ids tensor immediately to keep memory usage low
        del ids
        
    print("Fusing active voxel IDs...")
    unique_active_ids = torch.nonzero(active_mask).squeeze(1)
    
    # Free the mask as we don't need it anymore
    del active_mask
    
    print(f"Found {unique_active_ids.shape[0]} active voxels out of {res[0]*res[1]*res[2]} total possible voxels.")
    
    print("Building sparse voxel grid...")
    grid_vertices, voxels, _ = build_sparse_grid_from_active_voxels(
        unique_active_ids, 
        grid_min.tolist(), 
        grid_max.tolist(), 
        res
    )
    
    num_vertices = grid_vertices.shape[0]
    print(f"Allocated {num_vertices} vertices.")
    
    # Initialize TSDF volume arrays
    sdf = torch.ones(num_vertices, dtype=torch.float32, device='cuda').contiguous()
    weight = torch.zeros(num_vertices, dtype=torch.float32, device='cuda').contiguous()
    
    color = None
    if use_color:
        color = torch.zeros((num_vertices, 3), dtype=torch.float32, device='cuda').contiguous()
    
    print("Starting TSDF Volume Integration loop over sparse grid...")
    for batch in tqdm(loader, total=len(dataset)):
        depth = batch["depth"][0].cuda(non_blocking=True).contiguous()
        w2c = batch["w2c"][0].cuda(non_blocking=True).contiguous()
        intrinsics = batch["intrinsics"][0].cpu().contiguous()
        
        rgb = None
        if use_color:
            rgb = batch["color"][0].cuda(non_blocking=True).contiguous()
        
        # Fire the lightning-fast CUDA kernel
        single_view_volume_integral(
            grid_vertices=grid_vertices,
            sdf=sdf,
            weight=weight,
            depth_image=depth,
            extrinsics=w2c,
            intrinsics=intrinsics,
            color=color,
            color_image=rgb,
            trunc_margin=trunc_margin,
            mode=mode
        )
        
    print("Integration complete!")
    
    print("Extracting mesh using Marching Cubes...")
    # Extract the isosurface mesh directly on the GPU
    vertices, triangles, _, out_colors = marching_cubes(
        grid_vertices=grid_vertices,
        voxels=voxels,
        voxel_values=sdf,
        grid_colors=color,
        iso=0.0
    )
    
    return vertices, triangles, out_colors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PC2Mesh TSDF Volume Integration with conquer3d")
    parser.add_argument("--dataset_name", type=str, default="redwood", help="Name of the dataset (e.g., redwood)")
    parser.add_argument("--scene_name", type=str, default="apartment", help="Name of the scene (e.g., apartment)")
    parser.add_argument("--voxel_size", type=float, default=0.0056, help="Voxel size in meters")
    parser.add_argument("--trunc_margin", type=float, default=0.04, help="Truncation margin in meters")
    parser.add_argument("--mode", type=int, default=0, help="0 for Projective SDF, 1 for Euclidean SDF")
    parser.add_argument("--use_color", action="store_true", help="Enable color integration")
    parser.add_argument("--data_root", type=str, default="/home/koi/data", help="Root directory where datasets are stored")
    
    args = parser.parse_args()
    
    DATA_DIR = os.path.join(args.data_root, args.dataset_name, args.scene_name)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        sys.exit(1)
        
    verts, faces, colors = volume_integration_pc2mesh(
        dataset_name=args.dataset_name,
        scene_name=args.scene_name,
        data_dir=DATA_DIR,
        voxel_size=args.voxel_size,
        trunc_margin=args.trunc_margin,
        mode=args.mode,
        use_color=args.use_color
    )
    
    print(f"Extracted Mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")
    
    # Save the output to a structured directory
    out_dir = os.path.join("outputs", args.dataset_name, args.scene_name)
    os.makedirs(out_dir, exist_ok=True)
    
    output_path = os.path.join(out_dir, "mesh.ply")
    print(f"Exporting mesh to {output_path}...")
    
    # Trimesh expects float colors to be in [0, 1] or uint8 in [0, 255].
    # Since our integration maintains colors in [0, 255] float range, we must cast to uint8.
    vertex_colors = None
    if colors is not None:
        vertex_colors = torch.clamp(colors, 0, 255).to(torch.uint8).cpu().numpy()
    
    mesh = trimesh.Trimesh(
        vertices=verts.cpu().numpy(),
        faces=faces.cpu().numpy(),
        vertex_colors=vertex_colors
    )
    
    mesh.export(output_path)
    print("Done!")
