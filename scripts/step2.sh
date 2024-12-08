repo_dir="/path/to/TOTEM_for_EEG_code"

dataset_name="example"
python -m steps.STEP2_train_vqvae \
    +exp=train_vqvae \
    ++exp.save_dir="${repo_dir}/pipeline/step2_train_vqvae/${dataset_name}" \
    ++exp.vqvae_config.dataset=${dataset_name} \
    ++exp.vqvae_config.dataset_base_path="${repo_dir}/pipeline/step1_revin_x_data" \
    +logging=comet 