# python ./src/eval.py ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Cars/patch/clean/r0.02 mesh.threshold=0.005 experiment.name=data_full_bdry
# python ./src/eval.py ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/DeepFashion/patch/clean/r0.018 mesh.threshold=0.003 experiment.name=data_full_bdry

python ./src/eval.py ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Cars/patch/clean/r0.018/part1 mesh.threshold=0.003 mesh.is_cut=false experiment.name=data_full_bdry
# /data/nglm005/hjb/UDF/logs/train-smooth-w-noise/runs/2024-05-10_14-27-28/checkpoints/epoch_936.ckpt
# /data/nglm005/hjb/UDF/logs/train-full-w-noise/runs/2024-05-12_12-58-41/checkpoints/epoch_983.ckpt
# /data/nglm005/hjb/UDF/logs/train-smooth-bdry-w-noise/runs/2024-05-10_14-33-27/checkpoints/epoch_971.ckpt
# /data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt