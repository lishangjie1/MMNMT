#!/bin/bash


#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 10G


SHARD=5

file_name="train.en-ru.ru"
tlines=`awk 'END{print NR}' $file_name`
plines=`expr $tlines / $SHARD + 1`
split -l $plines $file_name "$file_name.split."

cnt=1
for file in `ls $file_name.split.*`
do
    mkdir -p part-$cnt #../../data-bin/para/data-shard-$cnt
    mv $file part-$cnt/$file_name #../../data-bin/para/data-shard-$cnt/$file_name
    cnt=`expr $cnt + 1`
done
