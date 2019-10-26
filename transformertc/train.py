# coding=utf-8
# Copyright 2019 Luis Rei
# MIT License

"""
Train function for Token level Classification with BERT
"""

import logging
import torch

from transformers import AdamW, WarmupLinearSchedule
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm, trange

from .evaluation import pytorch_evaluate


logger = logging.getLogger(__name__)


def set_seed(seed):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_steps_and_epochs(epochs, epoch_steps, max_steps):
    # if max_steps is passed, that is the number of total steps
    # and we need to calculate the number of epochs
    if max_steps > 0:
        total_steps = max_steps
        epochs = max_steps // epoch_steps + 1
    # if not, we calculate the total number of steps
    # based on the total number of epochs
    else:
        total_steps = epoch_steps * epochs
        epochs = epochs

    return total_steps, epochs


def log_train_start(total_steps, epoch_steps, epochs, batch_size):
    logger.info("***** Running training *****")
    logger.info("  Num steps/epoch = %d", epoch_steps)
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  batch size = %d", batch_size)
    logger.info("  Total optimization steps = %d", total_steps)


def get_optimizer_and_scheduler(model, lr, wdecay, adam_eps, warmup_steps,
                                total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': wdecay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                     t_total=total_steps)

    return optimizer, scheduler


def log_step(model,
             device,
             logging_steps,
             global_step,
             logging_loss,
             total_loss,
             scheduler,
             val_dataloader=None,
             label_list=None,
             no_tqdm=False):

    if logging_steps <= 0:
        # logging at every `logging_steps` is disabled
        return logging_loss

    if global_step % logging_steps != 0:
        # no logging at this step
        return logging_loss

    # Log metrics
    logger.info(f'Logging step: {global_step}')
    # if evaluating during training
    if val_dataloader is not None:
        pytorch_evaluate(model=model,
                         device=device,
                         dataloader=val_dataloader,
                         label_list=label_list,
                         prefix='val',
                         no_tqdm=no_tqdm)

    # normal logging: lr, loss
    lr = scheduler.get_lr()[0]
    logger.info(f'lr={lr}')
    avg_loss = (total_loss - logging_loss) / logging_steps
    logger.info(f'avg_loss={avg_loss}')
    return total_loss


def pytorch_train(model,
                  device,
                  dataloader,
                  epochs=4,
                  max_steps=0,
                  lr=5e-5,
                  wdecay=0.0,
                  warmup_steps=0,
                  adam_epsilon=1e-8,
                  max_grad_norm=1.0,
                  logging_steps=0,
                  labels=None,
                  val_dataloader=None,
                  no_tqdm=False):
    r"""Train the model with pytorch.
        Either args.num_train epochs or args.max_steps need to be passed.
        The latter takes priority if both are passed.

    Arguments:
        args.device: the device (e.g. gpu) where the model is
        args.max_steps (int, optional): maximum steps, defaults to
            epochs * steps_per_epoch
        args.num_train_epochs (int, optional): number of epochs to train for
        args.learning_rate
        args.wdecay
        args.warmup_steps
        args.adam_epsilon
        args.max_grad_norm
        args.no_tqdm (bool, optional): enable or disable TQDM
        args.seed (int, optional): random seed for python/numpy/torch
        args.checkpoint_dir (str): save model checkpoints to this directory
        args.logging_steps (int, optional): log every `logging_steps`

    """
    batch_size = dataloader.batch_size

    # calculate the total number of steps or the total number of epochs
    epoch_steps = len(dataloader)
    total_steps, epochs = get_steps_and_epochs(epochs,
                                               epoch_steps,
                                               max_steps)

    # optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model,
                                                       lr,
                                                       wdecay,
                                                       adam_epsilon,
                                                       warmup_steps,
                                                       total_steps)

    #
    # Train Cycle
    #
    log_train_start(total_steps, epoch_steps, epochs, batch_size)

    # train cycle variables
    global_step = 0     # counts total number of steps taken across epochs
    total_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()

    # Epoch iterator
    train_iterator = trange(int(epochs), desc="Epoch", disable=no_tqdm)
    for _ in train_iterator:
        # Step iterator
        epoch_iterator = tqdm(dataloader, desc="Step",
                              disable=no_tqdm)
        for step, batch in enumerate(epoch_iterator):
            model.train()

            batch = tuple(t.to(device) for t in batch)

            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': batch[3]
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)

            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # Log/Eval
            logging_loss = log_step(model=model,
                                    device=device,
                                    logging_steps=logging_steps,
                                    global_step=global_step,
                                    logging_loss=logging_loss,
                                    total_loss=total_loss,
                                    scheduler=scheduler,
                                    val_dataloader=val_dataloader,
                                    label_list=labels)

            # check if steps > total_steps (end of train)
            if global_step > total_steps:
                epoch_iterator.close()
                break

        # check if steps > total_steps (end of train)
        if global_step > total_steps:
            train_iterator.close()
            break

    return global_step, total_loss / global_step
