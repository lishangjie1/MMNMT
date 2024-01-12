#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
#print(os.environ)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
from fairseq_cli.train import cli_main
from fairseq_cli.generate_multiple import cli_main


if __name__ == "__main__":
    # build ext
    # if os.environ["LOCAL_RANK"] == "0":
    #     os.system("python setup.py build_ext --inplace")
    cli_main()
