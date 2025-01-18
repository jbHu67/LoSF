
# python ./src/eval_total.py \
#     ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
#     data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Additional/pc/clean \
#     mesh.threshold=0.005 \
#     mesh.is_cut=true \
#     experiment.name=data_full_bdry \
#     data.radius=0.01 \
#     paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Additional/output/ \
#     trainer.devices=[7]
python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Additional/pc/clean \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_full_bdry \
    data.radius=0.015 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Additional/output_cut/ \
    trainer.devices=[7]