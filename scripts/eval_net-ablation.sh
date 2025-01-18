# python ./src/extract_patch_feature.py

# python ./src/extract_patch_feature.py \
#     data.data_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/datasets/deepfashion/deepfashion_outlier/ \
#     paths.save_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/output/deepfashion_outlier/

# python ./src/extract_patch_feature.py \
#     data.data_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/datasets/shapenetcars/shapenetcars_noise/ \
#     paths.save_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/output/shapenetcars_noise/

# python ./src/extract_patch_feature.py \
#     data.data_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/datasets/shapenetcars/shapenetcars_outlier/ \
#     paths.save_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/output/shapenetcars_outlier/


# python ./src/eval_ablation.py \
#     ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation1/runs/2024-08-05_11-06-28/checkpoints/epoch_573.ckpt \
#     data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/cloth_noise/part2/ \
#     model.net._target_=src.models.components.ablation1_net.FeatureAttnLayer \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_full_bdry \
#     data.radius=0.018 \
#     paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation1 \
#     trainer.devices=[0]

# python ./src/eval_ablation.py \
#     ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation2/runs/2024-08-05_11-06-44/checkpoints/epoch_566.ckpt \
#     data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/cloth_noise/part1/ \
#     model.net._target_=src.models.components.ablation2_net.FeatureLayer \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_full_bdry \
#     data.radius=0.018 \
#     paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation2 \
#     trainer.devices=[0]

# python ./src/eval_ablation.py \
#     ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation1/runs/2024-08-05_11-06-28/checkpoints/epoch_573.ckpt \
#     data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/cloth_noise/part1/ \
#     model.net._target_=src.models.components.ablation1_net.FeatureAttnLayer \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_full_bdry \
#     data.radius=0.018 \
#     paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation1 \
#     trainer.devices=[0]

# python ./src/eval_ablation.py \
#     ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation2/runs/2024-08-05_11-06-44/checkpoints/epoch_566.ckpt \
#     data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/cloth_noise/part2/ \
#     model.net._target_=src.models.components.ablation2_net.FeatureLayer \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_full_bdry \
#     data.radius=0.018 \
#     paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation2 \
#     trainer.devices=[0]

python ./src/eval_ablation.py \
    ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation1/runs/2024-08-05_11-06-28/checkpoints/epoch_573.ckpt \
    data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/selected/cloth2/ \
    model.net._target_=src.models.components.ablation1_net.FeatureAttnLayer \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_full_bdry \
    data.radius=0.018 \
    paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation1 \
    trainer.devices=[0]

python ./src/eval_ablation.py \
    ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation2/runs/2024-08-05_11-06-44/checkpoints/epoch_566.ckpt \
    data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/selected/cloth2/ \
    model.net._target_=src.models.components.ablation2_net.FeatureLayer \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_full_bdry \
    data.radius=0.018 \
    paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation2 \
    trainer.devices=[0]

# python ./src/eval_ablation.py \
#     ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation1/runs/2024-08-05_11-06-28/checkpoints/epoch_573.ckpt \
#     data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/car_noise/part1/ \
#     model.net._target_=src.models.components.ablation1_net.FeatureAttnLayer \
#     mesh.threshold=0.005 \
#     mesh.is_cut=false \
#     experiment.name=data_full_bdry \
#     data.radius=0.018 \
#     paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation1 \
#     trainer.devices=[0]

# python ./src/eval_ablation.py \
#     ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation2/runs/2024-08-05_11-06-44/checkpoints/epoch_566.ckpt \
#     data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/car_noise/part2/ \
#     model.net._target_=src.models.components.ablation2_net.FeatureLayer \
#     mesh.threshold=0.005 \
#     mesh.is_cut=false \
#     experiment.name=data_full_bdry \
#     data.radius=0.018 \
#     paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation2 \
#     trainer.devices=[0]

# python ./src/eval_ablation.py \
#     ckpt_path=/home/cheese/Workzone/LoSF/CKPT/Network-ablation/train-ablation3/runs/2024-08-05_11-05-58/checkpoints/epoch_563.ckpt \
#     data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/net-ablation/selected/cloth/ \
#     model.net._target_=src.models.components.ablation3_net.FeatureAttnLayer \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_full_bdry \
#     data.radius=0.018 \
#     paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/net-ablation3/cloth_noise/ \
#     trainer.devices=[0]

