
set -e
DATA="test_data"
RAW_DATA="$DATA/raw_data"
CLEAN_DATA="$DATA/clean_data"
SPM_DATA="$DATA/spm_data"
BPE_DATA="$DATA/bpe_data"
DATA_BIN="$DATA/data-bin"

CODES=64000

# tools
MAIN="$PWD"
TOOLS_PATH="$MAIN/tools/preprocess_tools"
SPM_TRAIN=$TOOLS_PATH/spm_train
SPM_ENCODE=$TOOLS_PATH/spm_encode
NORM_PUNC=$TOOLS_PATH/normalize-punctuation.perl
REPLACE_UNICODE_PUNCT=$TOOLS_PATH/replace-unicode-punctuation.perl
REM_NON_PRINT_CHAR=$TOOLS_PATH/remove-non-printing-char.perl
INPUT_FROM_SGM=$TOOLS_PATH/input-from-sgm.perl

# clean tools
CLEAN_TOOLS=$MAIN/tools/clean_tools
DEDUP_MONO=$CLEAN_TOOLS/deduplicate_mono.py

###################################################################
# 1. normalize/clean raw data
mkdir -p $CLEAN_DATA

PREPROCESSING="$REM_NON_PRINT_CHAR"
lang_pairs=`ls $RAW_DATA`

for split in "train" "valid" "test"; do
    for lang_pair in $lang_pairs; do
        pair_arr=(${lang_pair//-/ })
        src_lang=${pair_arr[0]}
        tgt_lang=${pair_arr[1]}

        mkdir -p $CLEAN_DATA/$lang_pair
        cat $RAW_DATA/$lang_pair/$split.$src_lang | $PREPROCESSING > $CLEAN_DATA/$lang_pair/$split.$src_lang
        cat $RAW_DATA/$lang_pair/$split.$tgt_lang | $PREPROCESSING > $CLEAN_DATA/$lang_pair/$split.$tgt_lang
    done
done
###################################################################

# 2. tokenize
mkdir -p $SPM_DATA

# gather train data
for lang_pair in $lang_pairs; do
    pair_arr=(${lang_pair//-/ })
    src_lang=${pair_arr[0]}
    tgt_lang=${pair_arr[1]}

    cat $CLEAN_DATA/$lang_pair/train.$src_lang $CLEAN_DATA/$lang_pair/train.$tgt_lang
done > $SPM_DATA/spm_data

# deduplicate
python $DEDUP_MONO --rec-file $SPM_DATA/spm_data --out-file $SPM_DATA/spm_data.dedup

# train spm model
spm_train \
--normalization_rule_name identity \
--input $SPM_DATA/spm_data.dedup \
--model_prefix $SPM_DATA/spm \
--vocab_size ${CODES} \
--character_coverage 1.0 \
--model_type bpe

# encode spm_data to obtain vocabulary
spm_encode --model $SPM_DATA/spm.model --output_format piece < $SPM_DATA/spm_data.dedup > $SPM_DATA/spm_data.spm
fairseq-preprocess \
    --trainpref $SPM_DATA/spm_data.spm \
    --destdir $SPM_DATA \
    --workers 12 \
    --only-source \
    --dict-only

###################################################################
# 3. encode and binary
mkdir -p $BPE_DATA


for split in "train" "valid" "test"; do
    for lang_pair in $lang_pairs; do
        pair_arr=(${lang_pair//-/ })
        src_lang=${pair_arr[0]}
        tgt_lang=${pair_arr[1]}

        mkdir -p $BPE_DATA/$lang_pair

        spm_encode --model $SPM_DATA/spm.model \
            --output_format piece < $CLEAN_DATA/$lang_pair/$split.$src_lang > $BPE_DATA/$lang_pair/$split.$src_lang
        
        spm_encode --model $SPM_DATA/spm.model \
            --output_format piece < $CLEAN_DATA/$lang_pair/$split.$tgt_lang > $BPE_DATA/$lang_pair/$split.$tgt_lang
    done
done

mkdir -p $DATA_BIN
dict=$SPM_DATA/dict.txt

for lang_pair in $lang_pairs; do
    pair_arr=(${lang_pair//-/ })
    src_lang=${pair_arr[0]}
    tgt_lang=${pair_arr[1]}

    fairseq-preprocess \
        --source-lang ${src_lang} \
        --target-lang ${tgt_lang} \
        --srcdict $dict \
        --tgtdict $dict \
        --testpref $BPE_DATA/$lang_pair/test \
        --destdir $DATA_BIN \
        --workers 4 \

    fairseq-preprocess \
        --source-lang ${src_lang} \
        --target-lang ${tgt_lang} \
        --srcdict $dict \
        --tgtdict $dict \
        --validpref $BPE_DATA/$lang_pair/valid \
        --destdir $DATA_BIN \
        --workers 4 \
    
    fairseq-preprocess \
        --source-lang ${src_lang} \
        --target-lang ${tgt_lang} \
        --srcdict $dict \
        --tgtdict $dict \
        --trainpref $BPE_DATA/$lang_pair/train \
        --destdir $DATA_BIN \
        --workers 16 \

done
        




