#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 1
#SBATCH --mem 100G

tool_path="/nfs/users/lishangjie/fairseq/tools/fast_align/build/"
fast_align=$tool_path/fast_align


input_file=$1
$fast_align -i $input_file -d -o -v -s -I 3 > $input_file.align
