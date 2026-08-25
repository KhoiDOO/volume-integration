# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import math
import os, sys
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import *


def scalable_integrate_rgb_frames(path_dataset, intrinsic, config):
    poses = []
    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
    n_files = len(color_files)
    n_fragments = int(math.ceil(float(n_files) / \
            config['n_frames_per_fragment']))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))

    for fragment_id in range(len(pose_graph_fragment.nodes)):
        pose_graph_rgbd = o3d.io.read_pose_graph(
            join(path_dataset,
                 config["template_fragment_posegraph_optimized"] % fragment_id))

        for frame_id in range(len(pose_graph_rgbd.nodes)):
            frame_id_abs = fragment_id * \
                    config['n_frames_per_fragment'] + frame_id
            print(
                "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
                (fragment_id, n_fragments - 1, frame_id_abs, frame_id + 1,
                 len(pose_graph_rgbd.nodes)))
            rgbd = read_rgbd_image(color_files[frame_id_abs],
                                   depth_files[frame_id_abs], False, config)
            pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose,
                          pose_graph_rgbd.nodes[frame_id].pose)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            poses.append(pose)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if config["debug_mode"]:
        o3d.visualization.draw_geometries([mesh])

    mesh_name = join(path_dataset, config["template_global_mesh"])
    o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)

    traj_name = join(path_dataset, config["template_global_traj"])
    write_poses_to_log(traj_name, poses)


def run(config):
    print("integrate the whole RGBD sequence using estimated camera pose.")
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    scalable_integrate_rgb_frames(config["path_dataset"], intrinsic, config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Integrate scene")
    parser.add_argument("--path_dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--path_intrinsic", type=str, default="", help="Path to intrinsic json file")
    parser.add_argument("--depth_max", type=float, default=3.0, help="Maximum depth to truncate")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale factor")
    parser.add_argument("--tsdf_cubic_size", type=float, default=3.0, help="TSDF cubic size")
    parser.add_argument("--n_frames_per_fragment", type=int, default=100, help="Number of frames per fragment")
    parser.add_argument("--template_refined_posegraph_optimized", type=str, default="scene/refined_posegraph_optimized.json")
    parser.add_argument("--template_fragment_posegraph_optimized", type=str, default="fragments/frag_optimized_%03d.json")
    parser.add_argument("--template_global_mesh", type=str, default="scene/integrated.ply")
    parser.add_argument("--template_global_traj", type=str, default="scene/trajectory.log")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    config = {
        "path_dataset": args.path_dataset,
        "path_intrinsic": args.path_intrinsic,
        "depth_max": args.depth_max,
        "depth_scale": args.depth_scale,
        "tsdf_cubic_size": args.tsdf_cubic_size,
        "n_frames_per_fragment": args.n_frames_per_fragment,
        "template_refined_posegraph_optimized": args.template_refined_posegraph_optimized,
        "template_fragment_posegraph_optimized": args.template_fragment_posegraph_optimized,
        "template_global_mesh": args.template_global_mesh,
        "template_global_traj": args.template_global_traj,
        "debug_mode": args.debug_mode
    }
    
    run(config)
