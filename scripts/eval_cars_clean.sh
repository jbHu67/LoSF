python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Cars/pc/clean/undone/ \
    mesh.threshold=0.003 \
    mesh.is_cut=false \
    experiment.name=data_full_bdry \
    data.radius=0.018 \
    data.has_noise=false \
    data.noise_level=0.0 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Cars/output/clean/r0.018/out_mesh/undone \
    trainer.devices=[6]

