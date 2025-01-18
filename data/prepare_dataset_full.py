import os
import numpy as np
from tqdm import tqdm

data_smooth_folder = (
    "/home/cheese/Workzone/LoSF/TrainingDataset/uniform_sample/data_smooth/"
)
data_sharp_folder = "/home/cheese/Workzone/LoSF/TrainingDataset/data_sharp/"
folder_list = [
    data_smooth_folder,
    f"{data_sharp_folder}crease/",
    f"{data_sharp_folder}cusps/",
    f"{data_sharp_folder}saddle/",
    f"{data_sharp_folder}corner/",
]
data_full_folder = (
    "/home/cheese/Workzone/LoSF/TrainingDataset/uniform_sample/data_full/"
)
os.makedirs(data_full_folder, exist_ok=True)
data_id = 0
for folder in folder_list:
    file_list = []
    for file in os.listdir(folder):
        if file.endswith(".npz"):
            file_list.append(file)
    for file in tqdm(file_list):
        data = np.load(f"{folder}{file}")
        save_data = {}
        save_data["z_height"] = data["z_height"]
        save_data["verts"] = data["verts"]
        save_data["query"] = data["query"]
        np.savez(f"{data_full_folder}{data_id}.npz", **save_data)
        data_id += 1
print(f"Total number of data: {data_id}")
