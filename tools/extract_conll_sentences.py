#!/usr/bin/env python3
# coding=utf-8
# Copyright 2019 Luis Rei
# MIT License

"""
Simple script to extract sentences from a CONLL file.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--infile", default=None, type=str, required=True,
                        help="A file in CONLL format.")
    parser.add_argument("--outfile", default=None, type=str, required=True,
                        help="Save sentences to this file.")
    parser.add_argument("--max-sents", default=100, type=int,
                        help="Maximum number of sentences.")
    parser.add_argument("--min-len", default=10, type=int,
                        help="Maximum number of sentences.")

    args = parser.parse_args()
    return args


def read_sentences_from_conll_file(input_file, max_sents=0, min_len=0):
    examples = []
    with open(input_file, encoding="utf-8") as f:
        tokens = []
        for line in f:
            # check for new sequence (doc/sentence)
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    # end of setence, add example
                    if len(tokens) > min_len:
                        examples.append(tokens)
                    if max_sents > 0 and len(examples) >= max_sents:
                        break
                    tokens = []
                else:
                    # start of doc is usually
                    # -EMPTY -  [end of sentence and doc]
                    # DOCSTART  [start of doc]
                    # -EMPTY    [start of 1st sentence
                    # so there are multiple "empty" lines
                    pass
            else:
                # some "conll" use tabs intead of spaces
                # graciously covert them to the right format
                line = line.replace('\t', ' ')
                # split the line
                splits = line.split(" ")
                tokens.append(splits[0])
            # end of line
        # end of lines
        # if we have any tokens left, this means that there was no
        # final empty line at the end of the file but there was still
        # and example
        if tokens:
            if len(tokens) > min_len:
                examples.append(tokens)
    # end of with open
    return examples


def main():
    args = parse_args()

    sents = read_sentences_from_conll_file(args.infile,
                                           args.max_sents,
                                           args.min_len)

    with open(args.outfile, 'w') as f_out:
        for sent in sents:
            s = ' '.join(sent) + '\n'
            f_out.write(s)


if __name__ == '__main__':
    main()
