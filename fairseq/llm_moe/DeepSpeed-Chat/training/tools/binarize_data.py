import sys
import os
import argparse
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))

from utils.data import indexed_dataset

import time
import torch
import json
import pathlib
from utils.utils import load_hf_tokenizer_local

JSON_KEYS = {
    "text"
}


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')

    # Input files configuration
    group.add_argument('--data-prefix', type=str, required=True, help='Path Prefix to input JSON')

    group.add_argument("--seperator", type=str, default='\t', help="Seperator of input files")

    group.add_argument("--input-field-id", type=int, default=-1, help="Index of input field in input files")

    group = parser.add_argument_group(title='tokenizer')

    group.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                       help="Path to pretrained model or model identifier from huggingface.co/models")

    group = parser.add_argument_group(title='output data')

    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')

    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch')

    group.add_argument('--chunk-size', type=int, default=25,
                       help='Chunk size assigned to each worker process')

    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()

    return args

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text
class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.splitter = IdentitySplitter()

    # def encode(self, bucket_name, doc_id, json_line):
    def encode(self, args_tuple):
        doc_id, json_line = args_tuple
        data = json.loads(json_line)
        ids = {}

        # for key in self.args.json_keys:
        # text = data[key]
        text = data
        doc_ids = []
        for sentence in text:
            assert isinstance(sentence, list)

            sentence_ids = sentence

            if len(sentence_ids) > 0:
                doc_ids.append(sentence_ids)


        ids['text'] = doc_ids

        return ids, len(json_line)


def read_file(path: str, seperator='\t', input_field_id=-1):
    dir_name, base_prefix = os.path.dirname(path), os.path.basename(path)

    for fpath in pathlib.Path(dir_name).glob(f"{base_prefix}*"):

        fin = open(fpath, 'r', encoding='utf-8')

        for doc_id, line in enumerate(fin):
            if seperator in line:
                text = line.strip().split(seperator)[input_field_id]
                yield doc_id, text


def main():
    args = get_args()
    startup_start = time.time()

    fin = read_file(args.data_prefix, seperator=args.seperator, input_field_id=args.input_field_id)

    tokenizer = load_hf_tokenizer_local(args.pretrained_model_name_or_path)
    vocab_size = len(tokenizer)

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    encoded_docs = pool.imap(encoder.encode, fin, args.chunk_size)

    if not os.path.exists(os.path.dirname(args.output_prefix)):
        os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    print(f"Output prefix: {args.output_prefix}")

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    for key in JSON_KEYS:
        output_bin_files[key] = "{}_{}.bin".format(args.output_prefix, key)
        output_idx_files[key] = "{}_{}.idx".format(args.output_prefix, key)

        builders[key] = indexed_dataset.make_builder(output_bin_files[key], impl=args.dataset_impl,
                                                     vocab_size=vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):

        total_bytes_processed += bytes_processed

        for key, sentences in doc.items():

            if len(sentences) == 0:
                continue
            if key not in JSON_KEYS:
                continue

            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i / elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    for key in JSON_KEYS:
        builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    main()