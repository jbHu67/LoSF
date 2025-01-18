# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full-w-noise/runs/2024-05-12_12-58-41/checkpoints/epoch_983.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Outliers-analyze/pc \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.018 \
#     data.has_outliers=true \
#     data.outlier_ratio=0.1 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Outliers-analyze/output_0.1/ \
#     trainer.devices=[4]

# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full-w-noise/runs/2024-05-12_12-58-41/checkpoints/epoch_983.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Outliers-analyze/pc \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.018 \
#     data.has_outliers=true \
#     data.outlier_ratio=0.25 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Outliers-analyze/output_0.25/ \
#     trainer.devices=[4]
# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full-w-noise/runs/2024-05-12_12-58-41/checkpoints/epoch_983.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Outliers-analyze/pc \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.018 \
#     data.has_outliers=true \
#     data.outlier_ratio=0.5 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Outliers-analyze/output_0.5/ \
#     trainer.devices=[4]
# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full-w-noise/runs/2024-05-12_12-58-41/checkpoints/epoch_983.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Outliers-analyze/pc \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.018 \
#     data.has_outliers=true \
#     data.outlier_ratio=0.1 \
#     data.has_noise=false \
#     data.noise_level=0.0025 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Outliers-analyze/output_n+o/ \
#     trainer.devices=[4]

python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/pc \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_smooth \
    data.radius=0.025 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/output_0.025/ \
    trainer.devices=[4]
python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/pc \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_smooth \
    data.radius=0.03 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/output_0.03/ \
    trainer.devices=[4]
python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/pc \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_smooth \
    data.radius=0.005 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/output_0.005/ \
    trainer.devices=[4]