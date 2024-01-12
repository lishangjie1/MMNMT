

import sys


if __name__ == "__main__":
    src = sys.argv[1]
    tgt = sys.argv[2]
    
    with open(src, "r", encoding="utf-8") as fs, open(tgt, "r", encoding="utf-8") as ft, open("merged", "w", encoding="utf-8") as f:
        for line_src, line_tgt in zip(fs,ft):
            f.write(line_src.strip() + " ||| " + line_tgt.strip() + "\n")

