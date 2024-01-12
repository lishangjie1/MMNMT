#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -o clean_log

set -e
lang=$1
for lang in  "$lang" ; do
    echo "filter for $lang ..."
        src=train.zh-$lang.spm.$lang
        tgt=train.zh-$lang.spm.zh
        result_src=train.zh-$lang.spm.$lang.length_filter
        result_tgt=train.zh-$lang.spm.zh.length_filter
        python clean.py $src $tgt $result_src $result_tgt
        # rm $src
        # mv $result_src $src
        # rm $tgt
        # mv $result_tgt $tgt
    
done
