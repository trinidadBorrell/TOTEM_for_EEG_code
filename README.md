
# TOTEM for EEG code
Code and configs for the core model implementations used in the paper [Generalizability Under Sensor Failure: Tokenization + Transformers Enable More Robust Latent Spaces](https://arxiv.org/abs/2402.18546). Adapts the [original TOTEM implementation](https://arxiv.org/pdf/2402.16412) to EEG by implementing appropriate dataloaders and multivariate classification model. 

## Usage
1. Create a conda env with `env.yml`.
2. Convert your eeg data the expected format detailed in [Data format](#data-format) and put into your `<repo_dir>/data/` folder.
3. Edit the scripts with your `<repo_dir>` and `<dataset_name>`. 
4. Edit the `conf/` files to adjust exp, data processing, and model configurations. 
5. Run scripts for steps 1-4 in order using `bash ./scripts/stepX.sh`
    1. Step1 will take the multivariate EEG and convert it to normalized univariate EEG (using the ReVIN module) that are used as training trials for training the vq-vae. 
    2. Step2 will train the vq-vae using the data from Step1. 
        * You will need to set up comet for logging by copying `conf/logging/comet-template.yaml` to `conf/logging/comet.yaml` and adding your comet credentials.  
    3. Step3 will create the vq-vae tokenized multivariate EEG samples that will be used for downstream classification. 
    4. Step4 will train a xformer (transformer) classifier. 

## Data format
<details>
<summary> Detailed description: </summary>

* `dataset.csv`
    * First column is the index column which denote the timepoint in the recording.
    * The current implementation of `Dataset_EEG` assumes 128 channels
        * The example columns are for biosemi 128 channel device
    * `STI` is the label column
        * Values should be numbers representing the class as specified in `Dataset_EEG` `event_dict`
    * Units of EEG columns are in `uV` and preprocessing is done as specified in the [paper](https://arxiv.org/abs/2402.18546). 
* `dataset-split.csv`
    * First column is the index column which denote the timepoint in the recording.
        * Only the timepoints which mark the beginning of a new trial are kept in this file. 
    * `STI` is the label column
        * Values should be numbers representing the class as specified in `Dataset_EEG` `event_dict`
    * split is a column specifying the train test split assignments
        * Possible values: {train, val, test} 
</details>


<details>
<summary>Example data csv files:</summary>

`dataset.csv`
|  |    A1 |     A2 |    A3 |     A4 |     A5 |    A6 |    A7 |    A8 |     A9 |    A10 |    A11 |   A12 |    A13 |    A14 |    A15 |    A16 |    A17 |    A18 |    A19 |   A20 |    A21 |   A22 |   A23 |   A24 |   A25 |   A26 |   A27 |   A28 |   A29 |   A30 |   A31 |   A32 |    B1 |    B2 |    B3 |    B4 |    B5 |   B6 |    B7 |    B8 |    B9 |   B10 |   B11 |   B12 |   B13 |   B14 |   B15 |   B16 |   B17 |    B18 |   B19 |   B20 |    B21 |   B22 |    B23 |   B24 |   B25 |   B26 |    B27 |    B28 |    B29 |    B30 |   B31 |   B32 |    C1 |    C2 |    C3 |     C4 |     C5 |    C6 |    C7 |    C8 |    C9 |   C10 |   C11 |   C12 |   C13 |   C14 |   C15 |   C16 |    C17 |   C18 |   C19 |   C20 |   C21 |    C22 |   C23 |   C24 |    C25 |    C26 |   C27 |   C28 |    C29 |   C30 |   C31 |   C32 |     D1 |    D2 |     D3 |     D4 |     D5 |    D6 |    D7 |     D8 |     D9 |    D10 |    D11 |    D12 |    D13 |    D14 |    D15 |   D16 |    D17 |    D18 |    D19 |    D20 |    D21 |    D22 |    D23 |   D24 |    D25 |    D26 |    D27 |    D28 |   D29 |    D30 |   D31 |   D32 |   STI |
|-------------:|------:|-------:|------:|-------:|-------:|------:|------:|------:|-------:|-------:|-------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|------:|------:|-------:|------:|-------:|------:|------:|------:|-------:|-------:|-------:|-------:|------:|------:|------:|------:|------:|-------:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|------:|------:|-------:|-------:|------:|------:|-------:|------:|------:|------:|-------:|------:|-------:|-------:|-------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-------:|-------:|-------:|-------:|------:|-------:|------:|------:|------:|
|            0 |  0    |   0    | -0    |   0.01 |  -0    |  0    |  0    |  0    |   0    |   0    |   0    | -0    |  -0    |   0    |   0    |   0    |   0    |   0    |   0    |  0    |  -0    |  0.01 | -0    | -0    | -0.01 | -0    | -0    | -0    |  0.01 |  0.01 | -0    |  0    |  0    |  0    |  0    | -0    | -0    | 0    |  0    |  0    | -0    | -0    | -0    |  0    |  0    | -0    | -0    |  0    |  0    |   0    | -0    | -0    |   0    | -0    |   0    | -0    | -0.01 | -0.01 |   0    |   0    |   0    |  -0    |  0    | -0    |  0    | -0    |  0    |  -0    |  -0    |  0    |  0    |  0    | -0    | -0    |  0    |  0    |  0    |  0    |  0    |  0    |   0    |  0    | -0    | -0    |  0    |  -0    | -0    | -0    |   0    |   0    | -0    | -0    |   0    |  0    | -0    |  0    |   0.01 |  0    |  -0    |  -0    |  -0    | -0    | -0    |   0    |  -0.01 |  -0.01 |  -0    |  -0    |  -0    |   0    |   0    | -0    |   0    |   0    |  -0    |   0    |   0    |  -0    |  -0.01 | -0    |   0    |   0    |   0    |   0    |  0    |   0    |  0    | -0    |     0 |
|            1 | -1.6  | -35.99 |  2.29 | -29.54 |  -7.82 | -3.52 | -4.2  | -7.48 |  -5.39 |  -1.24 |  -2.87 | -2.12 |  -3.67 |   0.61 |  -1.41 | -12.01 |  -9.75 |  -3.76 |  -9.29 | -7.56 | -61.4  | -6.41 | -1.24 | -2.71 |  4.58 | -9.75 | -0.3  | -6.72 | -4.77 | 41.88 | -1.95 | -3.13 | -0.41 | -4.08 |  4.14 |  1.47 | 25.73 | 0.08 |  2.66 | 11.76 | -4.17 | -7.05 |  6.64 | -0.7  | -1.98 | 10.43 | -2.96 | -0.01 |  3.77 |   0.08 |  3.95 |  3.21 |   2.28 |  6.41 |  10.73 |  7.61 | -0.74 | 18.04 | -17.57 | -13.97 |  35.06 |  18.59 |  9.04 |  6.6  | -4.74 |  4.94 |  6.82 |   6.31 |   5.01 | -4.5  |  5.69 | 21.68 | 41.79 |  4.74 |  2.95 |  0.03 |  7.14 |  7.53 | 15.66 | 14.73 |  -0.66 |  4.89 |  1.8  |  2.29 |  2.71 |  -2.43 |  2.07 |  0.22 | -38.28 | -11.08 |  3.03 | 11.77 | -15.7  | 30.52 |  8.4  | -2.69 | -33.4  | -7.47 |  -5.32 | -14.82 | -24.94 | 57.53 | 73.82 |  42.32 |  24.04 | -21.78 | -12.06 |  -7.91 |  -5.64 |  -7.72 | -13.47 |  2.75 |  -3.08 |  -2.26 | -13.08 | -17.2  | -22.2  |  -2.36 |  15.9  | -2.39 | -12.93 | -18.67 | -10.09 |  -4.68 | -7.15 | -12.41 | -0.46 | -1.87 |     0 |
|            2 | -5.56 | -36.17 | -2.59 |  23.37 |  -9.78 | -8.41 | -8.28 | -8.91 |  -9.48 |   2.39 |   6.32 |  0.45 |  -1.05 |   0.66 |  -7.01 | -20.86 | -21.67 |  -8.31 |  -8.89 | -9.55 |   9.03 | 34.7  | -1.18 | -4.13 |  8.33 | -7.69 |  4.34 | -9.4  | -1.64 | 24.99 | -0.58 | -3.62 | -2.36 | -3.34 |  2.24 | -4.59 | 20.4  | 0.31 | -6.53 |  5.71 | -4.3  | -8.25 |  5.5  | -0.64 | -4.83 | 10.19 |  0.29 |  0.39 |  4.14 |   1.68 |  3.74 |  2.99 |   7.58 | 11.09 |  16.92 | 13.68 |  6.03 | 65.67 |   0.36 | -41.3  |  54.4  |  29.01 | 14.34 |  8.33 | -3.28 |  5.66 | 11.03 |  12.05 |   8.42 |  4.99 | -2.32 | 39.48 | 41.17 |  6.07 |  3.49 |  1.77 |  6.89 |  9.07 | 14.23 | 14.78 |  -0.04 |  4.22 |  5.14 |  4.45 |  4.9  |  -4.88 |  0.55 | -2.22 | -31.27 |  -7.15 |  7.76 |  3.98 | -23.93 | 15.23 | 24.37 |  9.81 | -32.04 | -8.79 | -10.91 | -18.49 | -33.9  | 71.95 | 44.52 |  -4.48 |  17.57 | -44.9  | -29.88 | -19.35 | -11.52 | -11.83 | -22.75 | -1.68 | -10.03 | -10.88 | -23.34 | -30.65 | -35.71 | -24.61 |  -1.32 | -6.48 | -18.47 | -23.99 | -15.5  |  22.84 | -8.37 | -11.33 |  1.95 | -0.75 |     0 |
|            3 | -7.11 | -16.91 | -5.81 |  -0.97 | -12.2  | -7.62 | -5.75 | -5.68 |  -9.96 | -10.65 | -18.99 | -7.38 | -10.08 | -10.02 | -15.13 |  -5.39 |  -5.37 |  -8.93 | -15.64 | -2.06 | 100.66 | 69.33 | -2.35 | -4.1  |  1.69 | -7.12 | 14.09 | -5.68 | -3.96 | 91.12 | -4.78 | -6.83 | -5.79 | -9.89 | -4.05 | -3.73 | 19.81 | 1.61 |  3.42 | 14.57 | -3.41 | -1.16 | 11.24 | -5.14 | -9.73 |  8    |  5.82 | -5.75 | -2.06 |  -6.4  | -2.95 | -6.77 | -10.26 | -0.59 |  -5.21 |  1.96 | 16.33 | 52.02 |  57.2  |  58.43 | -25.45 | -12.58 | -8.27 | -8.23 | -6.22 | -3.19 | -6.6  | -10.39 | -17.86 | 12.27 | 14.9  |  9.39 | 51.4  |  3.41 | -6.08 | -8.93 | -0.72 |  2.55 |  1.93 | 15.03 | -12.41 | -5.9  | -4.1  | -2.94 | -5.61 | -11.54 | -2.15 | -5.79 |  -5.68 | -17.17 | -5.06 | -1.34 | -26.28 | -6.3  | -6.73 |  2.29 | -11.7  | -4.17 | -12.12 | -26.76 | -33.13 | 53.59 |  6.4  | -44.68 | -11.72 | -12.07 | -15.5  |  -8.91 |  -9.09 | -10.32 |  -5.33 |  9.45 |  -5.16 |  -3.34 |  -8.29 |  -4.59 |  14.07 | -10.43 | -39.93 | -4.36 |  -5.11 |  -1.79 |  -1.81 | 110.36 |  5.28 |  -2.66 | -5.6  | -8.14 |     0 |
|            4 | -6.57 |  -5.34 | -6.3  |  26.85 | -11.39 | -8.28 | -5.12 | -8.08 | -14.77 |  -8.31 |  13.56 | -7.62 |  -9.57 |  -4.74 |  -6.45 | -15.6  | -18.74 | -13.65 | -14.41 | -7.22 |  56.43 | 60.44 | -6.56 | -4.48 | -5.32 | -9.21 | 58.9  | -7.63 | -1.76 | 77.02 | -8.79 | -5.93 | -5.33 | -8.28 | -7.49 | -6.32 | 14.79 | 0.82 |  1.13 |  9.1  | -8.88 |  1.05 | 12.49 | -6.5  | -9.31 |  1.14 |  7.53 | -5.74 | -5.32 | -10.57 | -1.3  | -5.14 |  -9.03 | -5.63 | -10.08 | -3.02 |  3.55 | 50.76 |  33.02 |  34.78 | -20.86 | -13.59 | -9.31 | -3.8  | -6.94 | -0.8  | -3.86 |  -6.41 |  -7.89 |  9.62 | 16.54 |  3.01 | 66.54 |  9.39 | -4    | -4.65 |  0.54 |  7.89 |  0.89 | 13.27 |  -7.71 | -2.83 |  1.84 |  0.79 | -0.67 |  -6.93 | -1.72 | -6.03 | -13.22 | -10.36 |  0.09 |  1.71 | -22.24 | -1.61 |  2.07 | 10.55 | -35.57 | -0.58 |  -9.58 | -16.55 | -21.11 | 57.74 | -2.71 | -28.81 |   6.32 | -17.83 |  -9.61 |  -8.4  |  -7.59 |  -8.58 |  -2.45 |  1.28 |  -5.55 |   0.5  |  -7.85 |  -3.75 |   5.8  | -23.28 | -35.7  | -5.49 | -12.6  |  -2.36 |  -4.14 |  93.6  | -3.58 |  -5.87 | -3.88 | -4.71 |     0 |
|            5 | -3.25 | -15.77 | -7.52 |  45.73 | -13.16 | -8.64 | -3.79 | -7.87 | -19.53 | -13.22 |   1.68 | -6.54 |  -8.06 |  -5.58 |  -6.68 |   0.76 |  -1.42 | -11.03 | -16.23 | -8.85 |  55.46 | 85.37 | -8.1  | -5.76 | 10.28 | -7.81 | 33.75 | -1.06 |  8.36 | 45.86 | -2.84 | -1.96 | -4.43 | -9.47 | -4.1  | -3.31 | 11.25 | 3.32 | -3.2  |  7.94 | -1.95 | -0.54 |  9.16 | -1.31 | -6.27 | 16.58 | 12.04 |  2.44 | -2.08 |  -9.01 |  3.62 | -2.81 |   1.54 | -2.32 |  -0.33 |  3.53 |  1.53 | 41.14 |  18.85 | -10.15 |   3.47 |   1.74 | -0.56 | -1.05 | -4.58 | -1.6  |  3.32 |   1.92 |   0.88 | 13.58 | -7.47 |  7    | 32.84 |  8.13 | -4.23 | -3.04 |  2.49 |  3.13 | -6.68 | 18.92 |  -8.2  | -1.82 |  0.14 |  1.21 |  1.1  |  -5.52 | -2.81 | -5.13 |  -8.45 |  -8.15 | -0.36 |  4.58 | -29.45 | -3.43 |  8.95 |  9.27 | -20.29 | -2.12 |  -9.2  | -21.44 | -11.91 | 62.83 | 17.46 |  -8.79 |  36.64 | -11.79 | -19.74 |  -9.77 | -13.05 | -12.44 |  -4.29 |  4.74 | -10.37 |  -9.52 | -21.04 | -27.43 | -50.24 |   0.2  | -17.15 | -7.97 | -14.88 | -18.7  | -13.04 |  42.66 | -2.44 |  -5.42 | -7.46 | -8.45 |     0 |

... and many more rows, one per timepoint. 

`dataset-split.csv`
|       | STI | split |
|-------|-----|-------|
| 10000 | 1.0 | test  |
| 20240 | 2.0 | test  |
| 30480 | 2.0 | train |
| 40720 | 1.0 | train |
| 50960 | 2.0 | train |
| 61200 | 4.0 | train |
| 71440 | 3.0 | val   |
| 81680 | 1.0 | train |
| 91920 | 1.0 | val   |

... and more depending on number of trials you have
</details>

## FAQ
<details>
<summary>What sampling rate should I save my data in?</summary>

The pipeline has been tested with sampling rates 256-4096Hz, and is agnostic to the underlying sampling rate. Some sampling rates may work better with the default window sizes (96 timepoints for VQVAE training, and 512 timepoints for classification modeling), depending on the nature of the task. Experimentation is encouraged! That said, it is important that the `dataset-split.csv` file is properly indexed to leverage the same sampling rate as `dataset.csv`.
</details>

## Citation
[TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis](https://arxiv.org/pdf/2402.16412)
```
@article{talukder_totem_2024,
	title = {TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis},
	issn = {2835-8856},
	shorttitle = {TOTEM},
	url = {https://openreview.net/forum?id=QlTLkH6xRC},
	journal = {Transactions on Machine Learning Research},
	author = {Talukder, Sabera J. and Yue, Yisong and Gkioxari, Georgia},
	year = {2024}
}
```

[Generalizability Under Sensor Failure: Tokenization + Transformers Enable More Robust Latent Spaces](https://arxiv.org/abs/2402.18546)
```
@article{chau2024generalizability,
  title={Generalizability Under Sensor Failure: Tokenization+ Transformers Enable More Robust Latent Spaces},
  author={Chau, Geeling and An, Yujin and Iqbal, Ahamed Raffey and Chung, Soon-Jo and Yue, Yisong and Talukder, Sabera},
  journal={arXiv preprint arXiv:2402.18546},
  year={2024}
}
```
