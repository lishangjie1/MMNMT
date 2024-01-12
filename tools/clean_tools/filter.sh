#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -o "bad_cases"
lang=$1
python filter.py --input_corpus merged --input_score merged.align --output_src out.zh --output_trg out.$lang
