python ./src/extract_patch_feature.py \
    data.data_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/datasets/general/clean/ \
    paths.save_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/output/general_clean/

python ./src/extract_patch_feature.py \
    data.data_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/datasets/general/clean/ \
    data.has_noise=true \
    data.noise_level=0.0025 \
    paths.save_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/output/general_noise/

python ./src/extract_patch_feature.py \
    data.data_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/datasets/general/clean/ \
    data.has_outliers=true \
    data.outlier_ratio=0.1 \
    paths.save_dir=/home/cheese/Workzone/LoSF/local_patch_analysis/local_patch_analysis/output/general_outlier/