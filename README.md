# MMNMT
## overview
Codes for the paper of EMNLP2023: "MMNMT: Modularizing Multilingual Neural Machine Translation with Flexibly Assembled MoE and Dense Blocks".

## Requirements
- python>=3.7
- pytorch>=1.09
- boto3
- iopath
- Fairseq (install from source code)
- sacrebleu < 2.0
- fairscale

## installment
`pip install -e ./`

`python setup.py build_ext --inplace`

## Data processing
Process:
1. BPE training with sentencepiece
2. BPE apply
3. Binarizing the data with `fairseq-preprocess`

datasets:
- OPUS-100: download data from opus website. (the scripts for preprocessing of the OPUS-100 data is in `process_data.sh` )

## Training

Train basic models on OPUS-100: `bash run_basic_model.sh`

Train MMNMT model on OPUS-100: `bash run_mmnmt.sh`

## Inference

Generate multiple language pairs: `bash run_generate_multi.sh`

## Model Evaluation

Please refer to calculate_bleu.sh for details.
