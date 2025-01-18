import argparse
import gc

import numpy as np
# pip install trimesh[all]
import trimesh


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                  for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def get_chamfer_dist(src_mesh_path, tgt_mesh_path, num_samples=30000):
    # Chamfer
    src_mesh = trimesh.load(src_mesh_path)
    tgt_mesh = trimesh.load(tgt_mesh_path)
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)  # src_surf_pts  has shape of (num_of_pts, 3)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    # src_tgt_dist = src_tgt_dist.mean()
    # tgt_src_dist = tgt_src_dist.mean()
    sqre_src_tgt_dist = np.square(src_tgt_dist)
    sqre_tgt_src_dist = np.square(tgt_src_dist)

    src_tgt_dist = np.mean(sqre_src_tgt_dist)
    tgt_src_dist = np.mean(sqre_tgt_src_dist)
    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

    thresh = 0.03
    print('pred2stl, min, max, mean', sqre_src_tgt_dist.min(), sqre_src_tgt_dist.max(), sqre_src_tgt_dist.mean(), \
          sqre_tgt_src_dist.min(), sqre_tgt_src_dist.max(), sqre_tgt_src_dist.mean())
    src_tgt_dist_w_thresh = np.mean(sqre_src_tgt_dist[sqre_src_tgt_dist < thresh])
    tgt_src_dist_w_thresh = np.mean(sqre_tgt_src_dist[sqre_tgt_src_dist < thresh])
    chamfer_dist_w_thresh = (src_tgt_dist_w_thresh + tgt_src_dist_w_thresh) / 2

    names = ['pred2stl', 'stl2pred', 'cd', 'cd_thresh']
    result = {}
    result[names[0]] = src_tgt_dist
    result[names[1]] = tgt_src_dist
    result[names[2]] = chamfer_dist
    result[names[3]] = chamfer_dist_w_thresh

    return result