"""
@File    :   prepare_dataset_sharp.py
@Time    :   2025/01/17 17:19:45
@Author  :   Jiangbei Hu 
@Version :   1.0
@Contact :   jbhu@dlut.edu.cn
@Desc    :   create the training data for the sharp patch dataset
"""

import os
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from tqdm import tqdm
from torch_geometric.nn import fps
import torch

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

type_list = {"crease": 5000, "cusps": 3000, "saddle": 3000, "corner": 4000}


save_dir = ""  # directory to save the sharp patch data
os.makedirs(save_dir, exist_ok=True)


def generate_data(i, save_dir, type):
    iter_num = 6
    if type == "crease":
        k = np.random.uniform(0.0001, 1)
        scale = np.random.uniform(0.2, 5)
        d = np.abs((k * X - Y)) / np.sqrt((1 + k**2))
        Z = 1 - scale * d
    elif type == "cusps":
        d = np.sqrt((X**2 + Y**2))
        scale = np.random.uniform(0.2, 5)
        Z = 1 - scale * d
    elif type == "saddle":
        d = np.max((np.abs(X), np.abs(Y)), axis=0)
        scale = np.random.uniform(0.01, 3)
        sign = X * Y
        sign_max = np.max(sign)
        transition_width = sign_max / 5
        sign_func = np.where(
            sign < -transition_width,
            -1,
            np.where(sign > transition_width, 1, sign / transition_width),
        )
        Z = 1 - scale * d * sign_func
    elif type == "corner":
        scale = np.random.uniform(0.2, 5)
        d = np.max((np.abs(X), np.abs(Y)), axis=0)
        Z = 1 - scale * d
    verts0 = [X, Y, Z]
    ray_samples = np.random.uniform(1 - radius, 1 + radius, iter_num)

    for j in range(iter_num):
        ray_z = ray_samples[j]
        query = [0.0, 0.0, ray_z]
        udf_value = np.abs(1 - ray_z)
        verts = verts0
        dist = np.sqrt(np.sum((np.array(verts).T - query) ** 2, axis=1))
        mask = dist <= radius
        verts_sel = np.array(verts).T[mask]
        # break if no points are selected
        if verts_sel.shape[0] == 0:
            continue
        if udf_value - dist.min() > 0.0001:
            continue
        if verts_sel.shape[0] >= upper_sample:
            # fps sampling
            ratio = upper_sample / verts_sel.shape[0]
            sel_idx = fps(torch.tensor(verts_sel), ratio=ratio)
            verts_sel = verts_sel[sel_idx]
        else:
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
        udf_value = udf_value * scale_factor
        id = i * iter_num + j
        data = {}
        data["z_height"] = udf_value
        data["verts"] = verts_sel
        data["query"] = query
        data["type"] = type
        # save data to npz file
        np.savez(os.path.join(save_dir, f"{id}.npz"), **data)


if __name__ == "__main__":
    num_cores = cpu_count()
    type_list = {"crease": 5000, "cusps": 3000, "saddle": 3000, "corner": 4000}
    type = "corner"
    save_dir_sub = os.path.join(save_dir, type)
    os.makedirs(save_dir_sub, exist_ok=True)
    Parallel(n_jobs=num_cores)(
        delayed(generate_data)(i, save_dir_sub, type)
        for i in tqdm(range(type_list[type]))
    )
