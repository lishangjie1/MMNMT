

import os
import sys
import re
from zhon.hanzi import punctuation 
import string

english_punc = string.punctuation
chinese_punc = punctuation
punc = english_punc + chinese_punc

def length_filter(x, y):
    len_x = len(x.strip().split())
    if len_x == 0:
        return True
    len_y = len(y.strip().split())
    if len_y > 2.5 * len_x or 2.5 * len_y < len_x:
        return True
    return False

def chinese_filter(strs):
    length = len(strs)+0.1
    i = 0
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fff':
            i += 1
    if i/length > 0.1:       
        return False
    return True
def repetition_filter(strs):
    words = strs.split()
    length = len(words)+0.1
    uni_words = set(words)
    if len(uni_words)/length < 0.3:       
        return True
    return False

def punctuation_filter(strs):
    length = len(strs)+0.1
    i = 0
    for _char in strs:
        if _char in punc:
            i += 1
    if i/length < 0.5:  
        return False
    return True
def unk_filter(strs):
    if "< un k >" in strs:
        return True
    return False
if __name__ == "__main__":
    src = sys.argv[1]
    tgt = sys.argv[2]
    rsrc = sys.argv[3]
    rtgt = sys.argv[4]
    cnt = 0
    unk_filtered = 0
    length_filtered = 0
    language_filtered = 0
    repetition_filtered = 0
    punc_filtered = 0
    noise_type = ["length_filter","unk_filter", "language_filter", "repetition_filtered", "punc_filtered"]
    noised_data = [open(f"noised_data/{noise}","w",encoding="utf-8") for noise in noise_type]
    with open(src,"r",encoding="utf-8") as f1, open(tgt,"r",encoding="utf-8") as f2, \
            open(rsrc,"w",encoding="utf-8") as f3, open(rtgt,"w",encoding="utf-8") as f4:
        for line_src, line_tgt in zip(f1,f2):
            line_src = line_src.strip()
            line_tgt = line_tgt.strip()
            
            cnt += 1
            if length_filter(line_src,line_tgt):
                noised_data[0].write(line_src +" ||| "+line_tgt + "\n")
                length_filtered += 1
                continue
            if unk_filter(line_tgt):
                noised_data[1].write(line_tgt + "\n")
                unk_filtered += 1
                continue
            if chinese_filter(line_tgt):
                noised_data[2].write(line_tgt + "\n")
                language_filtered += 1
                continue
            if repetition_filter(line_tgt):
                noised_data[3].write(line_tgt + "\n")
                repetition_filtered += 1
                continue
            if punctuation_filter(line_tgt):
                noised_data[4].write(line_tgt + "\n")
                punc_filtered += 1
                continue
            f3.write(line_src.strip() + "\n")
            f4.write(line_tgt.strip() + "\n")
    print("#####")
    print(f"Read in {cnt} sentences, unk filter {unk_filtered} sentences, length filter {length_filtered} sentences, language fileter {language_filtered} sentences, repetition filter {repetition_filtered} sentences, punctuation filter {punc_filtered} sentences")

    # for f in noised_data:
    #     f.close() 





