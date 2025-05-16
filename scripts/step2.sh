repo_dir="/home/triniborrell/home/projects/TOTEM_for_EEG_code"

dataset_name="a_dataset"

python -m steps.STEP2_train_vqvae \
    +exp=step2_train_vqvae \
    ++exp.save_dir="${repo_dir}/pipeline/step2_train_vqvae/${dataset_name}" \
    ++exp.vqvae_config.dataset=${dataset_name} \
    ++exp.vqvae_config.dataset_base_path="${repo_dir}/pipeline/step1_revin_x_data" \
    ++exp.vqvae_config.num_epochs=3 \
    +logging=comet 