"""
@File    :   prepare_dataset_smooth.py
@Time    :   2025/01/17 17:14:40
@Author  :   Jiangbei Hu 
@Version :   1.0
@Contact :   jbhu@dlut.edu.cn
@Desc    :   create the training data for the smooth patch dataset
"""

import os
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from tqdm import tqdm
from torch_geometric.nn import fps
import torch
import matplotlib


matplotlib.use("Agg")

total_num = 20000
upper_sample = 128  # maximum number of samples within a sphere
radius = 0.05

x = np.linspace(-1.2 * radius, 1.2 * radius, 200)
y = np.linspace(-1.2 * radius, 1.2 * radius, 200)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()

mask = X**2 + Y**2 <= radius**2
X = X[mask]
Y = Y[mask]

# range of k1 and k2
max_k = 40
min_k = -30
K1 = np.random.uniform(min_k, max_k, total_num)
K2 = np.random.uniform(min_k, max_k, total_num)


save_dir = ""  # directory to save the smooth patch data
os.makedirs(save_dir, exist_ok=True)


def generate_data(i, save_dir):
    k1 = K1[i]
    k2 = K2[i]
    Z = 0.5 * (k1 * X**2 + k2 * Y**2)
    verts = [X, Y, Z]
    ray_samples = np.random.uniform(0, radius, 5)
    for j in range(5):
        ray_z = ray_samples[j]
        query = [0.0, 0.0, ray_z]
        dist = np.sqrt(np.sum((np.array(verts).T - query) ** 2, axis=1))
        mask = dist <= radius
        verts_sel = np.array(verts).T[mask]

        if verts_sel.shape[0] == 0:
            continue  # break if no points are selected
        if ray_z - dist.min() > 0.0001:
            continue
        if verts_sel.shape[0] >= upper_sample:
            # fps sampling
            ratio = upper_sample / verts_sel.shape[0]
            sel_idx = fps(torch.tensor(verts_sel), ratio=ratio)
            verts_sel = verts_sel[sel_idx]
        else:  # padding to consistent with upper_sample
            padding_length = upper_sample - verts_sel.shape[0]
            repeat_number = padding_length // verts_sel.shape[0] + 1
            repeat_verts_sel = np.tile(verts_sel, (repeat_number, 1))
            # randomly re-order repeat_verts_sel
            np.random.shuffle(repeat_verts_sel)
            repeat_verts_sel = np.concatenate([verts_sel, repeat_verts_sel], axis=0)
            verts_sel = repeat_verts_sel[:upper_sample]
        # normalize to a sphere with radius 1
        verts_query = np.vstack([query, verts_sel])
        translation = np.mean(verts_query, axis=0)
        verts_query -= translation
        max_dist = np.max(np.linalg.norm(verts_query, axis=1))
        scale_factor = 1.0 / max_dist
        verts_query = verts_query * scale_factor
        query = verts_query[0]
        verts_sel = verts_query[1:]
        ray_z = ray_z * scale_factor
        id = i * 5 + j
        data = {}
        data["z_height"] = ray_z
        data["verts"] = verts_sel
        data["query"] = query
        data["curv"] = np.array([k1, k2])
        # save data to npz file
        np.savez(os.path.join(save_dir, f"{id}.npz"), **data)


if __name__ == "__main__":
    num_cores = cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(generate_data)(i, save_dir) for i in tqdm(range(total_num))
    )
