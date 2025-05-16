repo_dir="/home/triniborrell/home/projects/TOTEM_for_EEG_code"

dataset_name="a_dataset"

python -m steps.STEP3_save_classification_data \
    +preprocessing=step3_eeg \
    "++preprocessing.train_root_paths=['${repo_dir}/data']" \
    "++preprocessing.train_data_paths=['${dataset_name}.csv']" \
    "++preprocessing.test_root_paths=['${repo_dir}/data']" \
    "++preprocessing.test_data_paths=['${dataset_name}.csv']" \
    ++preprocessing.save_path="${repo_dir}/pipeline/step3_classification_data/${dataset_name}" \
    ++preprocessing.trained_vqvae_model_path="${repo_dir}/pipeline/step2_train_vqvae/${dataset_name}/checkpoints/final_model.pth" \
