#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 10G
input=$1
output=$2
lang=$3
perl normalize-punctuation.perl -l $lang < $input > $output
