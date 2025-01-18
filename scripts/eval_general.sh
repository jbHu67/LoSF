#need to re-run
python ./src/eval_total.py \
    ckpt_path=/home/cheese/Workzone/LoSF/CKPT/epoch_936.ckpt \
    data.data_dir=/home/cheese/Workzone/LoSF/EvalDataset/density/pc/clean/50k/ \
    mesh.threshold=0.003 \
    mesh.is_cut=false \
    experiment.name=data_full_bdry \
    data.radius=0.036 \
    data.has_noise=false \
    data.noise_level=0.0 \
    paths.output_dir=/home/cheese/Workzone/LoSF/outmesh/density/r_0_036_norot_time \
    trainer.devices=[0]
