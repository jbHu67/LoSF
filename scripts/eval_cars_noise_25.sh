python ./src/eval_total.py \
    ckpt_path=/data/nglm005/hjb/UDF/logs/train-full-w-noise/runs/2024-05-12_12-58-41/checkpoints/epoch_983.ckpt \
    data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/Cars/pc/clean/noise_undone/ \
    mesh.threshold=0.005 \
    mesh.is_cut=false \
    experiment.name=data_full \
    data.radius=0.018 \
    data.has_noise=true \
    data.noise_level=0.0025 \
    paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/Cars/output/noise_0.0025_0519/ \
    trainer.devices=[7]
