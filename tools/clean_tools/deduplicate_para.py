#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fileinput
import hashlib
import sys
import string
from multiprocessing import Pool

def get_hashes_and_lines(raw_lines):
    if isinstance(raw_lines, tuple):
        concat_line = raw_lines[0].strip() + raw_lines[1].strip()
        hash_src = hashlib.md5(concat_line.encode().strip()).hexdigest()
        return hash_src, raw_lines
    else:
        raise Exception("wrong input format")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pre-files-src", help="previous input files", default=None)
    parser.add_argument("--pre-files-tgt", help="previous input files", default=None)
    parser.add_argument("--rec-files-src", help="recent input source files")
    parser.add_argument("--rec-files-tgt", help="recent input source files")
    parser.add_argument("--out-file-src", help="recent output source files")
    parser.add_argument("--out-file-tgt", help="recent output source files")
    args = parser.parse_args()

    seen = set()
    if args.pre_files_src is not None:
        with open(args.pre_files_src, mode="r") as h_src, open(args.pre_files_tgt, mode="r") as h_tgt:
            pool = Pool(args.workers)
            results = pool.imap_unordered(get_hashes_and_lines, zip(h_src, h_tgt), 1000)
            for i, (hash, raw_lines) in enumerate(results):
                if hash not in seen:
                    seen.add(hash)
                if i % 1000000 == 0:
                    print(i, file=sys.stderr, end="", flush=True)
                elif i % 100000 == 0:
                    print(".", file=sys.stderr, end="", flush=True)


    with open(args.rec_files_src, mode="r") as h_src, open(args.rec_files_tgt, mode="r") as h_tgt, \
        open(args.out_file_src, mode="w") as w_src, open(args.out_file_tgt, mode="w") as w_tgt:
        pool = Pool(args.workers)
        results = pool.imap_unordered(get_hashes_and_lines, zip(h_src,h_tgt), 1000)
        for i, (hash, raw_lines) in enumerate(results):
            if hash not in seen:
                seen.add(hash)
                w_src.write(raw_lines[0].strip() + "\n")
                w_tgt.write(raw_lines[1].strip() + "\n")

            if i % 1000000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100000 == 0:
                print(".", file=sys.stderr, end="", flush=True)
    
    print(file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
