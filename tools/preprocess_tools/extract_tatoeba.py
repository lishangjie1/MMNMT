
import sys


if __name__ == "__main__":
    input_file = sys.argv[1]
    src = sys.argv[2]
    tgt = sys.argv[3]

    with open(input_file, "r", encoding="utf-8") as f, open(src, "w", encoding="utf-8") as fs, open(tgt, "w", encoding="utf-8") as ft:
        for line in f:
            splited = line.split("\t")
            src_line, tgt_line = splited[2].strip(), splited[3].strip()
            fs.write(src_line+"\n")
            ft.write(tgt_line+"\n")
