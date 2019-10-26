# coding=utf-8
# Copyright 2019 Luis Rei
# MIT License

"""
Inference (prediction) functions for Token level Classification with BERT
"""

import copy
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from .datatc import (convert_tokens_to_example, convert_example_to_features,
                     convert_features_to_dataset, ResultTC)


logger = logging.getLogger(__name__)


def features_from_tokens(idtokens,
                         tokenizer,
                         label2id,
                         ignore_label_id,
                         max_length):
    sid, tokens = idtokens
    example = convert_tokens_to_example(tokens, str(sid))
    features, _ = convert_example_to_features(example,
                                              label2id,
                                              tokenizer,
                                              max_length,
                                              ignore_label_id)

    return features


def extract_features(texts,
                     tokenizer,
                     label2id,
                     ignore_label_id,
                     max_length,
                     n_jobs=-1):
    """Creates a dataset at inference time.
    Args:
    """
    label2id = copy.deepcopy(label2id)
    label2id[''] = 0

    if n_jobs <= 0:
        n_jobs = cpu_count()

    logger.info(f'Feature extraction: {len(texts)} items on {n_jobs} jobs')

    all_features = []
    texts = enumerate(texts)
    extractor = partial(features_from_tokens,
                        tokenizer=tokenizer,
                        label2id=label2id,
                        ignore_label_id=ignore_label_id,
                        max_length=max_length)
    with Pool(processes=n_jobs) as pool:
        all_features = pool.map(extractor, texts)

    return all_features


def extract_bio(positions, tokens, preds, scores):
    results = []

    def make_entity(entity_tokens, entity_labels, entity_scores, start, end):
        lbl = entity_labels[0][2:]
        name = ' '.join(entity_tokens)
        r = ResultTC(name=name,
                     label=lbl,
                     token_start=start,
                     token_end=end,
                     score=np.mean(entity_scores))
        return r

    e_tokens = []
    e_labels = []
    e_scores = []
    start = 0
    end = 0
    for pos, token, label, score in zip(positions, tokens, preds, scores):
        if label.upper() == 'O':
            # end of an entity
            if len(e_tokens):
                r = make_entity(e_tokens, e_labels, e_scores, start, end)
                results.append(r)
                e_tokens, e_labels, e_scores = [], [], []
                continue
            # or just another O in a sequence of Os
            else:
                continue
        # we have an entity label
        else:
            # it's either a continuation or a new entity just after the current
            if len(e_tokens):
                # continuation
                if label[0].upper() == 'I' and label[2:] == e_labels[0][2:]:
                    e_tokens.append(token)
                    e_labels.append(label)
                    e_scores.append(score)
                    end = pos
                    continue
                # new entity just after the previous
                else:
                    r = make_entity(e_tokens, e_labels, e_scores, start, end)
                    results.append(r)
                    e_tokens, e_labels, e_scores = [], [], []
                    start = pos
                    end = pos
                    e_tokens.append(token)
                    e_labels.append(label)
                    e_scores.append(score)
            # it's a new entity and previous was either O or start of sentence
            else:
                start = pos
                end = pos
                e_tokens.append(token)
                e_labels.append(label)
                e_scores.append(score)
                continue

    # check if sentence ended on an entity
    if len(e_tokens):
        r = make_entity(e_tokens, e_labels, e_scores, start, end)
        results.append(r)

    return results


def extract_simple(positions, tokens, preds, scores):
    results = []

    for pos, token, label, score in zip(positions, tokens, preds, scores):
        r = ResultTC(name=token,
                     token_start=pos,
                     token_end=pos,
                     label=label,
                     score=score)
        results.append(r)

    return results


def convert_classification_to_result(texts,
                                     all_features,
                                     all_predictions,
                                     all_scores,
                                     id2label,
                                     task_format='BIO'):
    """Convert model predictions to a list of ``ResultTC``."""
    results = []
    iterator = None
    iterator = zip(texts, all_features, all_predictions, all_scores)

    for tokens, feats, preds, scores in iterator:
        # no need to check empty examples
        if len(feats.token_positions) == 0:
            results.append([])
            continue

        # mask predictions
        mask = np.array(feats.label_ids) >= 0
        preds = np.array(preds)[mask].tolist()
        scores = np.array(scores)[mask].tolist()
        # convert label ids to labels
        preds = [id2label[v] for v in preds]

        # ensure max token length
        length = feats.token_positions[-1] + 1
        tokens = tokens[:length]
        positions = list(range(0, length))

        # basic length assertion
        assert len(preds) == len(tokens), 'Length mismatch tokens != preds'

        result = None
        # simple token classification
        if task_format is None or task_format.upper() == 'SIMPLE':
            result = extract_simple(positions, tokens, preds, scores)
        elif task_format.upper() == 'BIO':
            result = extract_bio(positions, tokens, preds, scores)
        else:
            raise ValueError(f'Invalid task_format={task_format}')

        results.append(result)

    return results


def pytorch_classify(model, device, all_features, batch_size=None,
                     no_tqdm=False):

    dataset = convert_features_to_dataset(all_features,
                                          dataset_type='pytorch')
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batch_size,
                            drop_last=False)

    y_pred = []
    y_scores = []

    for batch in tqdm(dataloader, desc='Classifying', disable=no_tqdm):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            _, scores = outputs[:2]

            scores = scores.detach()

            # scores has shape (batch_size, sequence_length, num_labels)
            # get the predicted label ids via argmax
            # so we get (batch_size, seqlen) which should match labels
            preds = None
            scores = torch.nn.functional.softmax(scores, dim=2)
            scores, preds = torch.max(scores, dim=2)
            y_scores.append(scores.cpu().numpy())
            # preds is now (batch_size, sequence_lenght)

            # lets cpu the predictions and append them to our list
            y_pred.append(preds.cpu().numpy())

    # convert predictions to a list of lists
    # shape = (n_texts x max_seq_length) where n_texts = len(dataset)
    y_pred = np.concatenate(y_pred).tolist()
    y_scores = np.concatenate(y_scores).tolist()

    return y_pred, y_scores
