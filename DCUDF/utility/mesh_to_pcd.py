"""
Convert a Mesh file to a Point cloud file

"""

import argparse
import os
import trimesh


def mesh_to_pcd(
        mesh_path,
        pcd_path='default.ply',
        num_of_points=3000,
        save_pcd=True):
    mesh = trimesh.load(mesh_path)
    
    # Rescale the mesh to 1
    mesh.apply_scale(1/mesh.scale)

    sampled_pcd_points = trimesh.sample.sample_surface(mesh, num_of_points, seed=10)

    final_pcd = trimesh.points.PointCloud(sampled_pcd_points[0])

    if save_pcd:
        final_pcd.export(pcd_path)

    return final_pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mesh_path',
        required=True,
        help='The path to the mesh file')
    parser.add_argument(
        '--pcd_path',
        required=False,
        default='default.ply',
        help='The path to the new generated pcd file')
    parser.add_argument(
        '--number_of_points',
        required=False,
        default=3000,
        help='number of points in the new generated pcd file')

    arg = parser.parse_args()

    mesh_to_pcd(arg.mesh_path, arg.pcd_path, int(arg.number_of_points))
