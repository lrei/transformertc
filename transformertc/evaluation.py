# coding=utf-8
# Copyright (c) 2019 Luis Rei
# MIT License

"""
Evaluation function for Token level Classification with BERT
"""

import os
import logging

import torch
import numpy as np
from tqdm import tqdm
from seqeval.metrics import accuracy_score, f1_score
from seqeval.metrics import classification_report

logger = logging.getLogger(__name__)


def log_eval_start(eval_steps, batch_size, prefix=''):
    logger.info(f"***** Running evaluation: {prefix} *****")
    logger.info("  Num steps = %d", eval_steps)
    logger.info("  Batch size = %d", batch_size)


def log_eval_end(results, report, prefix=''):
    logger.info(f'***** Eval results: {prefix} *****')
    for key in results:
        logger.info("  %s = %s", key, str(results[key]))
    logger.info(report)


def save_eval(results, report, output_dir=None, prefix='val'):
    if not output_dir:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fname = prefix + '_' + 'eval_results.txt'
    output_file = os.path.join(output_dir, fname)

    with open(output_file, "w") as writer:
        for key in results:
            writer.write("%s = %s\n" % (key, str(results[key])))
        print(report, file=writer)


def pytorch_evaluate(model, device, dataloader, label_list,
                     output_dir=None, prefix='', no_tqdm=False):
    y_pred = []
    y_true = []
    id2label = {i: label for i, label in enumerate(label_list)}

    batch_size = dataloader.batch_size
    eval_steps = len(dataloader)

    # log start of evaluation
    log_eval_start(eval_steps, batch_size, prefix)

    eval_loss = 0.0
    nb_eval_steps = 0
    eval_iterator = tqdm(dataloader, desc="Evaluating",
                         disable=no_tqdm)
    for batch in eval_iterator:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, scores = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

            labels = inputs['labels'].detach()
            scores = scores.detach()

            # scores has shape (batch_size, sequence_length, num_labels)
            # get the predicted label ids via argmax
            # so we get (batch_size, seqlen) which should match labels
            scores = torch.argmax(scores, dim=2)
            # scores is now (batch_size, sequence_lenght)
            # attention and labels should be (batch_size, sequence_length)

            # we can now flatten things into a 1d array
            # which will be (batch x seqlen)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # remove padding from labels and their respective scores
            mask = labels >= 0
            scores = scores[mask]
            labels = labels[mask]

            # finally lets cpu the tensors and append them to our lists
            y_pred.append(scores.cpu().numpy())
            y_true.append(labels.cpu().numpy())

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    # convert predictions and true to a list of labels
    y_pred = np.concatenate(y_pred).tolist()
    y_true = np.concatenate(y_true).tolist()

    # convert label ids to text labels
    y_pred = [id2label[v] for v in y_pred]
    y_true = [id2label[v] for v in y_true]
    # assert len(y_true) == len(y_pred)

    results = {}
    results['loss'] = eval_loss
    results['acc'] = accuracy_score(y_true, y_pred)
    results['f1'] = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    log_eval_end(results, report)
    save_eval(results, report, output_dir, prefix)

    return results
