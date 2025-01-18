
python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/NumSamples/pc/large_r/ \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_full_bdry \
    data.radius=0.018 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/NumSamples/pc/sparse_small_r/output/ \
    trainer.devices=[5]
# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/pc \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.02 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/output_0.02/ \
#     trainer.devices=[7]
# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/pc \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_smooth \
#     data.radius=0.015 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/output_0.015/ \
#     trainer.devices=[7]

# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/pc \
#     mesh.threshold=0.005 \
#     mesh.is_cut=false \
#     experiment.name=data_smooth \
#     data.radius=0.025 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/output_0.025/ \
#     trainer.devices=[4]
# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/pc \
#     mesh.threshold=0.005 \
#     mesh.is_cut=false \
#     experiment.name=data_smooth \
#     data.radius=0.03 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Radius-analyze/output_0.03/ \
#     trainer.devices=[4]