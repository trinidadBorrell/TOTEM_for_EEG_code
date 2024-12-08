
# TOTEM for EEG code
Code and configs for the core model implementations used in the paper [Generalizability Under Sensor Failure: Tokenization + Transformers Enable More Robust Latent Spaces](https://arxiv.org/abs/2402.18546). Adapts the [original TOTEM implementation](https://arxiv.org/pdf/2402.16412) to EEG by implementing appropriate dataloaders and multivariate classification model. 

## Usage
1. Create conda env with `env.yml`
2. Obtain eeg data in the format in the [Data prep](#data-prep) section below. 
3. Run scripts for steps 1-4 in order 
    1. Step1 will take the multivariate EEG and convert it to normalized univariate EEG that are used as training trials for training the vq-vae. 
    2. Step2 will train the vqvae using the data from Step1. 
    3. Step3 will create the vq-vae tokenized multivariate EEG samples that will be used for downstream classification. 
    4. Step4 will train a transformer classifier 

## Data prep
* For EEG, make sure your data is formatted with columns as such: 
    * dataset.csv:,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19,A20,A21,A22,A23,A24,A25,A26,A27,A28,A29,A30,A31,A32,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24,D25,D26,D27,D28,D29,D30,D31,D32,STI
        * First column is the index column which denote the timepoint in the recording
        * The current implementation of Dataset_EEG assumes 128 channels (the current headings are for biosemi 128 channel device)
        * STI is the label column (values should be numbers representing the class as specified in Dataset_EEG event_dict)
        * Units are in uV and preprocessing is done as specified in the paper. 
    * dataset-split.csv: ,STI,split
        * First column is the index that corresponds to the beginning of this trial with label specified
        * STI is the label column (values should be numbers representing the class as specified in Dataset_EEG event_dict)
        * split is a column with these values: {train, val, test} 
