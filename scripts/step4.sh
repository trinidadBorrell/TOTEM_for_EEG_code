repo_dir="/home/triniborrell/home/projects/TOTEM_for_EEG_code"

dataset_name="a_dataset"

python -m steps.STEP4_train_xformer \
    +exp=step4_train_xformer \
    ++exp.classifier_config.data_path="${repo_dir}/pipeline/step3_classification_data/${dataset_name}" \
    ++exp.classifier_config.checkpoint_path="${repo_dir}/pipeline/step4_train_xformer" \
    ++exp.classifier_config.exp_name=${dataset_name} \
    +logging=comet 