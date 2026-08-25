import os
import torch
import numpy as np

# Import the base dataset
from conquer3d.data.dataset.redwood import RedWood

# Import Open3D pipeline steps that were brought to the root folder
import make_fragments
import register_fragments
import refine_registration

class OptimizedRedwood(RedWood):
    """
    OptimizedRedwood wraps the raw Redwood dataset and transparently runs the Open3D
    Pose Graph Optimization pipeline (make -> register -> refine) on initialization.
    It then overrides the camera poses with the fully optimized trajectory, ensuring
    all TSDF integration receives perfectly drift-free camera tracking.
    """
    def __init__(self, data_root: str, scene_name: str = "", load_color: bool = True, depth_max: float = 3.0,
                 convert_rgb_to_intensity: bool = False, convert_rgb_to_intensity_type: str = 'weighted'):
        # Initialize the base dataset which will load paths, intrinsics, and the initial raw poses.
        super().__init__(data_dir=data_root, scene_name=scene_name, load_color=load_color, depth_max=depth_max,
                         convert_rgb_to_intensity=convert_rgb_to_intensity, convert_rgb_to_intensity_type=convert_rgb_to_intensity_type)
        
        # Build the exact Open3D pipeline config dict based on our argparse defaults
        config = {
            "path_dataset": self.data_dir,
            "path_intrinsic": "",
            "depth_max": depth_max,
            "depth_scale": 1000.0,
            "depth_diff_max": 0.07,
            "tsdf_cubic_size": 3.0,
            "n_frames_per_fragment": 100,
            "n_keyframes_per_n_frame": 5,
            "folder_fragment": "fragments/",
            "folder_scene": "scene/",
            "template_fragment_posegraph": "fragments/frag_%03d.json",
            "template_fragment_posegraph_optimized": "fragments/frag_optimized_%03d.json",
            "template_fragment_pointcloud": "fragments/frag_%03d.ply",
            "template_global_posegraph": "scene/posegraph.json",
            "template_global_posegraph_optimized": "scene/posegraph_optimized.json",
            "template_refined_posegraph": "scene/refined_posegraph.json",
            "template_refined_posegraph_optimized": "scene/refined_posegraph_optimized.json",
            "template_global_traj": "scene/trajectory.log",
            "voxel_size": 0.05,
            "global_registration": "ransac",
            "icp_method": "color",
            "python_multi_threading": True,
            "debug_mode": False,
            "preference_loop_closure_registration": 5.0,
            "preference_loop_closure_odometry": 0.1
        }
        
        # The ultimate output of the Open3D pose graph optimization is this trajectory file.
        self.optimized_log_path = os.path.join(self.data_dir, config["template_global_traj"])
        
        # Check if the optimization has already been performed. If not, trigger the pipeline.
        if not os.path.exists(self.optimized_log_path):
            print("==========================================================")
            print("Optimized trajectory not found! Triggering Open3D Pipeline")
            print("==========================================================")
            
            # Step 1: Make Fragments (Integrates mini-chunks of ~100 frames where drift is negligible)
            print("\n--- Step 1: Make Fragments ---")
            make_fragments.run(config)
            
            # Step 2: Register Fragments (Uses ICP and Fast Global Registration to stitch fragments)
            print("\n--- Step 2: Register Fragments ---")
            register_fragments.run(config)
            
            # Step 3: Refine Registration (Global Pose Graph Optimization across the whole scene)
            print("\n--- Step 3: Refine Registration ---")
            refine_registration.run(config)
            
            print("\nOpen3D Pose Optimization Complete!")
        else:
            print(f"Loaded existing optimized trajectory from: {self.optimized_log_path}")
            
        # Parse the newly created (or pre-existing) optimized trajectory using the base class method.
        # This overwrites the raw SLAM tracking from `apartment.log` with the perfect rigid poses.
        # Consequently, `__getitem__` will now inherently return these pristine `c2w` and `w2c` matrices.
        self.poses = self._parse_log_file(self.optimized_log_path)
