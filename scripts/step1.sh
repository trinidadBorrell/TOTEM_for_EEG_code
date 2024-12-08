repo_dir="/path/to/TOTEM_for_EEG_code"

dataset_name="example"

python -m steps.STEP1_save_revin_xdata_for_vqvae \
    +preprocessing=step1_eeg \
    "++preprocessing.root_paths=['${repo_dir}/data']" \
    "++preprocessing.data_paths=['${dataset_name}.csv']" \
    ++preprocessing.save_path="${repo_dir}/pipeline/step1_revin_x_data/${dataset_name}"
