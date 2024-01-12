
import os
import sys
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plot

count_line = "./mrasp_count_line"
count = {}
with open(count_line, "r") as f:
    for line in f:
        lang, cnt = line.strip().split()
        lang = lang.lower()
        langpair = f"{lang}-en"
        count[langpair] = int(cnt)

        src, tgt = langpair.split('-')
        reverse_lp = f"{tgt}-{src}"
        count[reverse_lp] = int(cnt)


def get_bleu(result_dir):
    result = {}
    result['en2any'], result['any2en'] = {}, {}
    any2en, any2en_num = 0, 0
    en2any, en2any_num = 0, 0

    scale_names = ['high', 'medium', 'low', 'extre_low']
    scales = [1e7, 1e6, 1e5, 0]
    any2en_bleu, any2en_cnt = [0] * len(scale_names), [0] * len(scale_names)
    en2any_bleu, en2any_cnt = [0] * len(scale_names), [0] * len(scale_names)

    for dir_name in os.listdir(result_dir):
        dir_path = os.path.join(result_dir, dir_name)
        if os.path.isdir(dir_path):
            src, tgt = dir_name.split('-')
            if src in ignore_languages or tgt in ignore_languages:
                continue
            output = os.popen(f"bash calculate_bleu_mrasp.sh {result_dir} {src} {tgt}")
            bleu = float(output.readlines()[0].strip())

            langpair = f"{src}-{tgt}"
            if src.startswith('en'):
                result['en2any'][langpair] = bleu
                en2any += bleu
                en2any_num += 1
                for scale_idx, scale in enumerate(scales):
                    if count[langpair] >= scale:
                        en2any_bleu[scale_idx] += bleu
                        en2any_cnt[scale_idx] += 1
                        break
            else:
                result['any2en'][langpair] = bleu
                any2en += bleu
                any2en_num += 1

                for scale_idx, scale in enumerate(scales):
                    if count[langpair] >= scale:
                        any2en_bleu[scale_idx] += bleu
                        any2en_cnt[scale_idx] += 1
                        break
    
    any2en = any2en / any2en_num if any2en_num > 0 else 0
    en2any = en2any / en2any_num if en2any_num > 0 else 0
    result['any2en_avg'] = any2en
    result['en2any_avg'] = en2any

    for scale_idx, scale_name in enumerate(scale_names):
        result[f'any2en_{scale_name}'] = any2en_bleu[scale_idx] / any2en_cnt[scale_idx]
        result[f'en2any_{scale_name}'] = en2any_bleu[scale_idx] / en2any_cnt[scale_idx]


    print(result)
    print(any2en_cnt, en2any_cnt)
    # resources divide

gen_dir = "/mnt/nas/users/lsj/moe/generate_dir/"
model_name="moe_model_mrasp_init_2_encdec_"


ignore_languages = ["mt", "eo"]
get_bleu(gen_dir + model_name)
    
# ex-low af be gu my