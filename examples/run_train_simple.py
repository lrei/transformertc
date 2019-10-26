#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2019 Luis Rei
# MIT License

"""
Train script for Token level Classification with BERT
Creates and evaluates a model.
"""


import argparse
import logging

import torch
from transformers import BertConfig

# import sys
# sys.path.insert(0, "../")
from transformertc import BertTC
from transformertc.datatc import DataBunchLoaderTC
from transformertc.evaluation import pytorch_evaluate


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys())
     for conf in (BertConfig, )),
    ())


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--data-dir", default=None, type=str, required=True,
                        help="The train/dev/test data files (CONLL format).")
    parser.add_argument("--cached-data-dir", default=None, type=str,
                        required=False,
                        help="Where to cache the data.")

    # pretrained model to load
    parser.add_argument("--model-name-or-path", default=None, type=str,
                        required=True,
                        help="Path to pre-trained model "
                        "or shortcut name selected in the list: " +
                        ", ".join(ALL_MODELS))

    #
    # Model config
    #
    parser.add_argument("--max-seq-length", default=128, type=int,
                        help="The maximum sequence length after tokenization.")
    parser.add_argument("--format", default='BIO', type=str,
                        help="Token Classification Format.")

    # outputs:  model save, checkpoints, data cache, pretrained cache, eval...
    parser.add_argument("--model-save-dir", default=None, type=str,
                        required=True,
                        help="The directory where the model will be saved.")
    parser.add_argument("--cache-dir", default="", type=str,
                        help="Cache location for downloaded pre-trained "
                        "models (via transformers).")
    parser.add_argument("--checkpoint-dir", default="", type=str,
                        help="Where to store checkpoints during training.")

    #
    # Train parameters
    #
    parser.add_argument("--train-batch-size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test-batch-size", default=1, type=int,
                        help="Batch size for test.")

    parser.add_argument("--learning-rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--wdecay", default=0.0, type=float,
                        help="Weight decay.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup-steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    # epochs /steps
    parser.add_argument("--num-epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")

    # log / checkpoint
    parser.add_argument("--no-tqdm", action='store_true',
                        help='disable TQDM progress bars')

    # debug options
    parser.add_argument("--no-cuda", action="store_true",
                        help="Don't use CUDA")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Device
    if torch.cuda.is_available() and not args.no_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # Setup logging
    log_format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    logging.basicConfig(format=log_format,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    # processor, labels
    bunch_loader = DataBunchLoaderTC(data_dir=args.data_dir,
                                     slen=args.max_seq_length,
                                     backend='pytorch',
                                     cached_data_dir=args.cached_data_dir)
    labels = bunch_loader.get_labels()
    logger.info(labels)

    #
    # Load the pretrained model
    #
    model = BertTC.create_from_pretrained(args.model_name_or_path,
                                          labels=labels,
                                          max_seq_length=args.max_seq_length,
                                          task_format=args.format)
    model.to(args.device)

    #
    # Load Data
    #
    bunch_loader.tokenizer = model.tokenizer
    # train
    train_dataloader = bunch_loader.get_train(args.train_batch_size)

    # test
    test_dataloader = bunch_loader.get_test(args.test_batch_size)

    #
    # Train
    #
    pbar = not args.no_tqdm
    model.finetune(dataloader=train_dataloader,
                   epochs=args.num_epochs,
                   lr=args.learning_rate,
                   wdecay=args.wdecay,
                   warmup_steps=args.warmup_steps,
                   adam_epsilon=args.adam_epsilon,
                   progressbar=pbar)

    logger.info('\nTraining Finished!\n')

    #
    # Evaluation
    #
    pytorch_evaluate(model=model.model,
                     device=model.device,
                     dataloader=test_dataloader,
                     label_list=labels,
                     output_dir=args.model_save_dir,
                     prefix='test',
                     no_tqdm=args.no_tqdm)

    #
    # Save
    #
    model.save_pretrained(args.model_save_dir)


if __name__ == '__main__':
    main()
