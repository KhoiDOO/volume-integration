# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/make_fragments.py

import math
import multiprocessing
import os, sys
import numpy as np
import open3d as o3d

from open3d_example import *
from optimize_posegraph import optimize_posegraph_for_fragment

# check opencv python package
with_opencv = initialize_opencv()
if with_opencv:
    from opencv_pose_estimation import pose_estimation


def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                           with_opencv, config):
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], True,
                                        config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], True,
                                        config)

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = config["depth_diff_max"]
    if abs(s - t) != 1:
        if with_opencv:
            success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                    target_rgbd_image,
                                                    intrinsic, False)
            if success_5pt:
                [success, trans, info
                ] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]


def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                             t - sid,
                                                             trans,
                                                             info,
                                                             uncertain=False))

            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 \
                    and t % config['n_keyframes_per_n_frame'] == 0:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s - sid, t - sid, trans, info, uncertain=True))
    o3d.io.write_pose_graph(
        join(path_dataset, config["template_fragment_posegraph"] % fragment_id),
        pose_graph)


def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        print(
            "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
            (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))
        rgbd = read_rgbd_image(color_files[i_abs], depth_files[i_abs], False,
                               config)
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config):
    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id, n_fragments,
        join(path_dataset,
             config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name = join(path_dataset,
                    config["template_fragment_pointcloud"] % fragment_id)
    o3d.io.write_point_cloud(pcd_name,
                             pcd,
                             format='auto',
                             write_ascii=False,
                             compressed=True)


def process_single_fragment(fragment_id, color_files, depth_files, n_files,
                            n_fragments, config):
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            
    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)

    posegraph_name = join(config["path_dataset"], config["template_fragment_posegraph"] % fragment_id)
    if not os.path.exists(posegraph_name):
        make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files,
                                    depth_files, fragment_id, n_fragments,
                                    intrinsic, with_opencv, config)
    else:
        print(f"Fragment {fragment_id} posegraph already exists. Skipping make_posegraph.")

    posegraph_opt_name = join(config["path_dataset"], config["template_fragment_posegraph_optimized"] % fragment_id)
    if not os.path.exists(posegraph_opt_name):
        optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    else:
        print(f"Fragment {fragment_id} optimized posegraph already exists. Skipping optimize_posegraph.")

    pcd_name = join(config["path_dataset"], config["template_fragment_pointcloud"] % fragment_id)
    if not os.path.exists(pcd_name):
        make_pointcloud_for_fragment(config["path_dataset"], color_files,
                                     depth_files, fragment_id, n_fragments,
                                     intrinsic, config)
    else:
        print(f"Fragment {fragment_id} pointcloud already exists. Skipping make_pointcloud.")


def run(config):

    print("making fragments from RGBD sequence.")
    os.makedirs(join(config["path_dataset"], config["folder_fragment"]), exist_ok=True)

    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    n_files = len(color_files)
    n_fragments = int(
        math.ceil(float(n_files) / config['n_frames_per_fragment']))

    if config["python_multi_threading"] is True:
        max_workers = min(max(1, multiprocessing.cpu_count() - 1), n_fragments)
        # Prevent over allocation of open mp threads in child processes
        os.environ['OMP_NUM_THREADS'] = '1'
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(processes=max_workers) as pool:
            args = [(fragment_id, color_files, depth_files, n_files,
                     n_fragments, config) for fragment_id in range(n_fragments)]
            pool.starmap(process_single_fragment, args)
    else:
        for fragment_id in range(n_fragments):
            process_single_fragment(fragment_id, color_files, depth_files,
                                    n_files, n_fragments, config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Make fragments from RGBD sequence")
    parser.add_argument("--path_dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--path_intrinsic", type=str, default="", help="Path to intrinsic json file")
    parser.add_argument("--depth_max", type=float, default=3.0, help="Maximum depth to truncate")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale factor")
    parser.add_argument("--depth_diff_max", type=float, default=0.07, help="Maximum depth difference for Odometry")
    parser.add_argument("--tsdf_cubic_size", type=float, default=3.0, help="TSDF cubic size")
    parser.add_argument("--n_frames_per_fragment", type=int, default=100, help="Number of frames per fragment")
    parser.add_argument("--n_keyframes_per_n_frame", type=int, default=5, help="Number of keyframes per frame for loop closure")
    parser.add_argument("--folder_fragment", type=str, default="fragments/", help="Folder to save fragments")
    parser.add_argument("--template_fragment_posegraph", type=str, default="fragments/frag_%03d.json")
    parser.add_argument("--template_fragment_posegraph_optimized", type=str, default="fragments/frag_optimized_%03d.json")
    parser.add_argument("--template_fragment_pointcloud", type=str, default="fragments/frag_%03d.ply")
    parser.add_argument("--disable_multi_threading", action="store_true", help="Disable python multi-threading")
    parser.add_argument("--preference_loop_closure_odometry", type=float, default=0.1, help="Preference loop closure odometry")
    
    args = parser.parse_args()
    
    config = {
        "path_dataset": args.path_dataset,
        "path_intrinsic": args.path_intrinsic,
        "depth_max": args.depth_max,
        "depth_scale": args.depth_scale,
        "depth_diff_max": args.depth_diff_max,
        "tsdf_cubic_size": args.tsdf_cubic_size,
        "n_frames_per_fragment": args.n_frames_per_fragment,
        "n_keyframes_per_n_frame": args.n_keyframes_per_n_frame,
        "folder_fragment": args.folder_fragment,
        "template_fragment_posegraph": args.template_fragment_posegraph,
        "template_fragment_posegraph_optimized": args.template_fragment_posegraph_optimized,
        "template_fragment_pointcloud": args.template_fragment_pointcloud,
        "python_multi_threading": not args.disable_multi_threading,
        "preference_loop_closure_odometry": args.preference_loop_closure_odometry
    }
    
    run(config)
