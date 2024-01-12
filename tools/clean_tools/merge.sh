#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 1
#SBATCH --mem 10G

src=$1
tgt=$2
python merge.py $src $tgt
