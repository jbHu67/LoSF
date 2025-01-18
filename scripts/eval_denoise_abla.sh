python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-smooth-w-noise/runs/2024-05-10_14-27-28/checkpoints/epoch_936.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/pc/sphere/ \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_smooth \
    data.radius=0.018 \
    data.has_noise=false \
    data.noise_level=0.0 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/output_clean/ \
    trainer.devices=[7]

python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-smooth-w-noise/runs/2024-05-10_14-27-28/checkpoints/epoch_936.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/pc/sphere/ \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_smooth \
    data.radius=0.018 \
    data.has_noise=true \
    data.noise_level=0.0025 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/output_0.0025/ \
    trainer.devices=[7]
python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-smooth-w-noise/runs/2024-05-10_14-27-28/checkpoints/epoch_936.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/pc/sphere/ \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_smooth \
    data.radius=0.018 \
    data.has_noise=true \
    data.noise_level=0.005 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/output_0.005/ \
    trainer.devices=[7]

# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-smooth-w-noise/runs/2024-05-19_08-17-33/checkpoints/epoch_905.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/pc/ \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.018 \
#     data.has_noise=true \
#     data.noise_level=0.0025 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/output/out_mesh_noise_wo_denoise/ \
#     trainer.devices=[7]

# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-smooth-w-noise/runs/2024-05-10_14-27-28/checkpoints/epoch_936.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/pc/ \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.018 \
#     data.has_outliers=true \
#     data.outlier_ratio=0.1 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/output/out_mesh_outlier/ \
#     trainer.devices=[7]

# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-smooth-w-noise/runs/2024-05-19_08-17-33/checkpoints/epoch_905.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/pc/ \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.018 \
#     data.has_outliers=true \
#     data.outlier_ratio=0.1 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Denoise-abla/output/out_mesh_outlier_wo_denoise/ \
#     trainer.devices=[7]