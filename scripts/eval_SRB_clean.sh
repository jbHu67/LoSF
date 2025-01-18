# python ./src/eval_total.py ckpt_path=/data/nglm005/hjb/UDF/logs/train-full_bdry-w-noise/runs/2024-05-12_13-01-58/checkpoints/epoch_988.ckpt data.data_dir=/data/nglm005/hjb/UDF/EvalDataset/DeepFashion/pc/clean/ mesh.threshold=0.004 experiment.name=data_full_bdry data.radius=0.018 data.has_noise=true data.noise_level=0.0025 paths.output_dir=/data/nglm005/hjb/UDF/EvalDataset/DeepFashion/output/noise_0.0025/

python ./src/eval_total.py \
    ckpt_path=/home/cheese/Workzone/LoSF/CKPT/epoch_988.ckpt \
    data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/SRB/L2/ \
    mesh.threshold=0.005 \
    mesh.is_cut=true \
    experiment.name=data_full_bdry \
    data.radius=0.018 \
    data.has_noise=false \
    data.noise_level=0.0 \
    paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/SRB_no/ \
    trainer.devices=[0]