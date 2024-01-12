import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--input_corpus", type=str, default='')
parser.add_argument("--input_score", type=str, default='')
parser.add_argument("--output_src", type=str, default='')
parser.add_argument("--output_trg", type=str, default='')
args = parser.parse_args()

file_input_corpus = open(args.input_corpus, 'r', encoding='utf-8')
file_input_score = open(args.input_score, 'r', encoding='utf-8')
file_output_src = open(args.output_src, 'w', encoding='utf-8')
file_output_trg = open(args.output_trg, 'w', encoding='utf-8')

i = 0
j = 0
align_rate = 0.4
align_score = -9.0

for src_trg, alignment in zip(file_input_corpus, file_input_score):
    i += 1
    pair = src_trg.strip().split("|||")
    if len(pair) != 2:
        print('error')
        print(pair)
        print(alignment)
        j += 1
        continue
    src, trg = pair
    src, trg = src.strip(), trg.strip()
    align, score = alignment.strip().split("|||")
    score = float(score)

    align_token = [x.split('-')[0] for x in align.strip().split()]
    src_len = len(src.split())
    trg_len = len(trg.split())
    align_len = len(set(align_token))

    if src_len == 0 or trg_len == 0:
        j += 1
        continue
    if align_len / src_len > align_rate and score / trg_len > align_score:
        file_output_src.write(src + '\n')
        file_output_trg.write(trg + '\n')
    else:
        print('pass')
        print(pair)
        print(alignment.strip())
        print("average score: ",score / trg_len)
        print("cover length: ",align_len/src_len)
        j += 1
        continue

    if i % 10000 == 0:
        print(i)
        print(j)

file_input_corpus.close()
file_input_score.close()
file_output_src.close()
file_output_trg.close()

# python -u select_align.py --input_corpus --input_score --output_src  --output_trg
# --input_corpus  源语言句子 ||| 目标语言句子
# --input_score  对齐 0-1 1-2 ||| 对齐得分

