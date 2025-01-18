# python ./src/eval_total.py ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/DeepFashion/pc/clean/ mesh.threshold=0.004 experiment.name=data_full_bdry data.radius=0.018 data.has_noise=true data.noise_level=0.0025 paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/DeepFashion/output/noise_0.0025/

python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-full-w-noise/runs/2024-05-12_12-58-41/checkpoints/epoch_983.ckpt\
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Cars/pc/clean/outlier_undone \
    mesh.threshold=0.005 \
    mesh.is_cut=false \
    experiment.name=data_smooth \
    data.radius=0.018 \
    data.has_outliers=true \
    data.outlier_ratio=0.1 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Cars/output/outlier_0.1/ \
    trainer.devices=[7]