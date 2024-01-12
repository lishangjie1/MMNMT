



import re
import sys
from bpe.bpe import BPE
inp, out = sys.argv[1], sys.argv[2]
src, tgt = sys.argv[3], sys.argv[4]
codes = "/mnt/nas/users/lsj/moe/mrasp2/mrasp_data/mRASP/experiments/test/vocab/codes.bpe.32000"
bpe = BPE(codes)
with open(inp, "r") as f, open(out, "w", encoding="utf-8") as f1:

    for line in f:
        if tgt == 'zh' or tgt == 'ja':
            line = re.sub(' ', '', line)
            line = re.sub('@@', '@@ ', line)
        detok = bpe.decode(line.strip())
        f1.write(detok.strip() + "\n")