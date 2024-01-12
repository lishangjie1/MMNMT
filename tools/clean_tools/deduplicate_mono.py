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


def get_hashes_and_lines(raw_line):
    if isinstance(raw_line, str):
        hash = hashlib.md5(raw_line.encode().strip()).hexdigest()
        return hash, raw_line
    else:
        raise Exception("wrong input")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pre-file", help="previous input files", default=None)
    parser.add_argument("--rec-file", help="recent input files")
    parser.add_argument("--out-file", help="recent output files")
    args = parser.parse_args()

    seen = set()
    if args.pre_file is not None:
        with open(args.pre_file, mode="r") as h:
            pool = Pool(args.workers)
            results = pool.imap_unordered(get_hashes_and_lines, h, 1000)
            for i, (hash, raw_line) in enumerate(results):
                if hash not in seen:
                    seen.add(hash)
                if i % 1000000 == 0:
                    print(i, file=sys.stderr, end="", flush=True)
                elif i % 100000 == 0:
                    print(".", file=sys.stderr, end="", flush=True)


    with open(args.rec_file, mode="r") as h, open(args.out_file, mode="w") as w:
        pool = Pool(args.workers)
        results = pool.imap_unordered(get_hashes_and_lines, h, 1000)
        for i, (hash, raw_line) in enumerate(results):
            if hash not in seen:
                seen.add(hash)
                w.write(raw_line.strip() + "\n")

            if i % 1000000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100000 == 0:
                print(".", file=sys.stderr, end="", flush=True)
    
    print(file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
