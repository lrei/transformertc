#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2019 Luis Rei
# MIT License

"""A script that loads a trained model (fine tuned) and runs inference.

 - Loads a pretrained, pre-finetuned model;
 - Loads a file with line delimited setences (pre-tokenized);
 - Runs classification;
 - Saves results to an output file;

"""

import argparse
import logging
import time

import torch

from transformertc import BertTC


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--target", default=None, type=str, required=True,
                        help="A file with line delimited tokenized sentences.")
    parser.add_argument("--output", default=None, type=str, required=False,
                        help="Save output to this file.")

    # Model options
    parser.add_argument("--model-path", default=None, type=str, required=True,
                        help="Path to pre-trained model.")
    parser.add_argument("--batch-size", default=0, type=int,
                        help="Batch size for training.")

    # Debug options
    parser.add_argument("--no-cuda", action="store_true",
                        help="Don't use CUDA")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Setup CUDA usage
    if torch.cuda.is_available() and not args.no_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # Setup logging
    log_format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    logging.basicConfig(format=log_format,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    # Load the model
    model = BertTC.from_pretrained(args.model_path)
    model.to(device=args.device)

    # Load the data
    # 1 tokenized sentence per line
    logger.info('Reading sentences')
    sentences = open(args.target).readlines()
    sentences = [x.strip().split() for x in sentences]

    # inference
    logger.info('Classifying sentences')
    start = time.time()
    results = model.classify(texts=sentences,
                             batch_size=args.batch_size,
                             n_jobs=-1,
                             progressbar=True)
    end = time.time()
    elapsed = int(end - start)
    logger.info(f'Elapsed Time = {elapsed}s')

    # save results
    logger.info('Saving results')
    with open(args.output, 'w') as f_out:
        for sentence_results in results:
            for r in sentence_results:
                f_out.write(r.to_txt_string() + ' ')
            f_out.write('\n')


if __name__ == '__main__':
    main()
