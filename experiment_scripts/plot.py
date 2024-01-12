
import os
import sys
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plot

count_line = "/mnt/nas/users/lsj/moe/moe_data/opus_dest/clean_data/count_line"
count = {}
with open(count_line, "r") as f:
    for line in f:
        langpair, cnt = line.strip().split()
        count[langpair] = int(cnt)

        src, tgt = langpair.split('-')
        reverse_lp = f"{tgt}-{src}"
        count[reverse_lp] = int(cnt)


def get_bleu(result_dir):
    result = {}
    result['en2any'], result['any2en'] = {}, {}
    any2en, any2en_num = 0, 0
    en2any, en2any_num = 0, 0

    any2en_bleu, any2en_cnt = [0, 0, 0], [0, 0, 0]
    en2any_bleu, en2any_cnt = [0, 0, 0], [0, 0, 0]

    for dir_name in os.listdir(result_dir):
        dir_path = os.path.join(result_dir, dir_name)
        if os.path.isdir(dir_path):
            src, tgt = dir_name.split('-')

            output = os.popen(f"bash calculate_bleu.sh {result_dir} {src} {tgt}")
            bleu = float(output.readlines()[0].strip())

            langpair = f"{src}-{tgt}"
            if src.startswith('en'):
                result['en2any'][langpair] = bleu
                en2any += bleu
                en2any_num += 1
                if count[langpair] > 500000:
                    en2any_bleu[0] += bleu
                    en2any_cnt[0] += 1
                elif count[langpair] > 200000:
                    en2any_bleu[1] += bleu
                    en2any_cnt[1] += 1
                else:
                    en2any_bleu[2] += bleu
                    en2any_cnt[2] += 1
                
            else:
                result['any2en'][langpair] = bleu
                any2en += bleu
                any2en_num += 1

                if count[langpair] > 500000:
                    any2en_bleu[0] += bleu
                    any2en_cnt[0] += 1
                elif count[langpair] > 200000:
                    any2en_bleu[1] += bleu
                    any2en_cnt[1] += 1
                else:
                    any2en_bleu[2] += bleu
                    any2en_cnt[2] += 1
    
    any2en = any2en / any2en_num if any2en_num > 0 else 0
    en2any = en2any / en2any_num if en2any_num > 0 else 0
    result['any2en_avg'] = any2en
    result['en2any_avg'] = en2any

    result['any2en_high'] = any2en_bleu[0] / any2en_cnt[0]
    result['any2en_medium'] = any2en_bleu[1] / any2en_cnt[1]
    result['any2en_low'] = any2en_bleu[2] / any2en_cnt[2]

    result['en2any_high'] = en2any_bleu[0] / en2any_cnt[0]
    result['en2any_medium'] = en2any_bleu[1] / en2any_cnt[1]
    result['en2any_low'] = en2any_bleu[2] / en2any_cnt[2]


    print(result)
    print(any2en_cnt, en2any_cnt)
    # resources divide

gen_dir = "/mnt/nas/users/lsj/moe/generate_dir/"

model_name="moe_model_mix_init_encdec_straight_0.75_93k"
get_bleu(gen_dir + model_name)
    
