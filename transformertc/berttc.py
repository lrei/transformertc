# -*- coding: utf-8 -*-
# Copyright (c) 2019 Luis Rei
# MIT License
"""BertTC class: BERT for token classification tasks.

For examples on how to use this class see the `examples` directory.
"""

import os
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizer

from .configtc import ConfigTC
from .classification import extract_features, pytorch_classify
from .classification import convert_classification_to_result
from .train import pytorch_train
from .datatc import ResultTC

from typing import List


class BertTC(object):
    r"""BertTC class: BERT for token classification tasks.

    This class allows:
        * loading pretrained and/or fine-tuned BERT models;
        * fine-tuning (really, training) models;
        * using fine-tuned models for classification (inference).

    It acts primarly as a wrapper around a transformer model, its config
    object, and tokenizer. Put togetherwith a handfull of useful functions.
    Namely save/load and fine-tune/classify.

    Attributes:
        config (BertConfig): pretained BertConfig object from transformers.
        configtc (ConfigTC): ConfigTC object.
        tokenizer (BertTokenizer): pretrained BertTokenizer from transformers.
        model (BertForTokenClassification): pretrained BERT model with the
            token classification layers added in (but not necessarily trained).
    """

    def __init__(self,
                 config: BertConfig,
                 configtc: ConfigTC,
                 tokenizer: BertTokenizer,
                 model: BertForTokenClassification):
        self.model = model
        self.config = config
        self.configtc = configtc
        self.tokenizer = tokenizer
        """Sets the attributes."""

    def save_pretrained(self, save_directory: str) -> None:
        """Save to a given directory."""
        # Create model save dir
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save config
        self.config.save_pretrained(save_directory)
        self.configtc.save_pretrained(save_directory)

        # Save model
        self.model.save_pretrained(save_directory)

        # Save the tokenizer
        self.tokenizer.save_pretrained(save_directory)

    def to(self, device):
        """Send model to a specific device."""
        self.device = device
        self.model.to(self.device)

    @classmethod
    def from_pretrained(cls, model_path):
        """Load from a given path."""
        # Load config
        config = BertConfig.from_pretrained(model_path)
        configtc = ConfigTC.from_pretrained(model_path)

        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path,
                                                  do_lower_case=False,
                                                  do_basic_tokenize=False)

        # Load model
        model = BertForTokenClassification.from_pretrained(model_path,
                                                           config=config)
        return cls(config, configtc, tokenizer, model)

    @classmethod
    def create_from_pretrained(cls,
                               model_name_or_path,
                               labels,
                               max_seq_length=0,
                               task_format='BIO'):
        num_labels = len(labels)

        config = BertConfig.from_pretrained(model_name_or_path,
                                            num_labels=num_labels)

        configtc = ConfigTC(labels=labels,
                            max_seq_length=max_seq_length,
                            task_format=task_format)

        tokenizer = BertTokenizer.from_pretrained(model_name_or_path,
                                                  do_lower_case=False,
                                                  do_basic_tokenize=False)

        model = BertForTokenClassification.from_pretrained(model_name_or_path,
                                                           config=config)

        return cls(config, configtc, tokenizer, model)

    def classify(self,
                 texts: List[List[str]],
                 batch_size: int = None,
                 n_jobs: int = -1,
                 progressbar: bool = False) -> List[List[ResultTC]]:
        """Classifiy a list of tokenized texts with this model.

        Args:
            texts (:obj:`list` of :obj:`list` of :obj:`str`): list of
                (word) tokenized documents (e.g. sentences). Example:
                :code:`[['This', 'is', '1'], ['And', 'this', 'is', '2']]`.
            batch_size (int): size of batches to use. Defaults to None which
                will try to use `len(texts)` as the batch_size.
            n_jobs (int): number of threads/processes to use when converting
                `texts` to features (i.e. ``InputFeaturesTC``). Defaults to
                `-1` which means a number equal to the number of CPU cores.
            progressbar (bool): show a progressbar (via TQDM) for the
                classification progress.

        Returns:
            A list of lists of ``ResultTC`` corresponding to the list of
            texts.
        """
        labels = self.configtc.labels
        task_format = self.configtc.task_format
        max_seq_length = self.configtc.max_seq_length

        label2id = {lbl: i for i, lbl in enumerate(labels)}
        id2label = {i: lbl for i, lbl in enumerate(labels)}
        no_tqdm = not progressbar

        # if no batch size is provided, we do a single batch
        if not batch_size:
            batch_size = len(texts)

        # guard against empty texts
        texts = [text for text in texts if len(text) > 0]

        # ignore label
        ignore_label_id = torch.nn.CrossEntropyLoss().ignore_index

        # extract features
        all_features = extract_features(texts=texts,
                                        tokenizer=self.tokenizer,
                                        label2id=label2id,
                                        ignore_label_id=ignore_label_id,
                                        max_length=max_seq_length,
                                        n_jobs=n_jobs)
        # guard against empty texts again
        all_features = [feats for feats in all_features
                        if len(feats.token_positions) > 0]
        # classify
        preds, scores = pytorch_classify(model=self.model,
                                         device=self.device,
                                         all_features=all_features,
                                         batch_size=batch_size,
                                         no_tqdm=no_tqdm)
        # convert to results
        results = convert_classification_to_result(texts=texts,
                                                   all_features=all_features,
                                                   all_predictions=preds,
                                                   all_scores=scores,
                                                   id2label=id2label,
                                                   task_format=task_format)

        return results

    def finetune(self,
                 dataloader,
                 epochs: int = 4,
                 lr: float = 5e-5,
                 wdecay: float = 0.0,
                 warmup_steps: int = 0,
                 adam_epsilon: float = 1e-8,
                 progressbar: bool = False) -> None:
        """Fine-tune pretrained model on a TC task.

        Arguments:
            dataloader (DataLoader): a pytorch dataloader.
            epochs (int): number of epochs to fine-tune for.
            lr (float): the learning rate.
            wdecay (float): weight decay.
            warmup_steps (int): number of steps to run linear warmup for.
            adam_epsilon (float): epsilon parameter for Adam optimizer.
            progressbar (bool): use TQDM progress bar during fine tuning.
        """

        no_tqdm = not progressbar
        pytorch_train(model=self.model,
                      device=self.device,
                      dataloader=dataloader,
                      epochs=epochs,
                      max_steps=0,
                      lr=lr,
                      wdecay=wdecay,
                      warmup_steps=warmup_steps,
                      adam_epsilon=adam_epsilon,
                      max_grad_norm=1.0,
                      logging_steps=0,
                      labels=self.configtc.labels,
                      val_dataloader=None,
                      no_tqdm=no_tqdm)

        return None
