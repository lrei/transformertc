# coding=utf-8
# Copyright (c) 2019 Luis Rei
# MIT License

"""
Data loading functions for Token level Classification with BERT
Reads data in the CONLL format.
"""
import os
import csv
import copy
import json
import logging

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset

from typing import Tuple, List, Dict, Sequence, TypeVar, Any
ExampleAttType = TypeVar('InputExampleTCAttribute', str, List[str])

logger = logging.getLogger(__name__)


class InputExampleTC(object):
    """A single training/test example for token classification.

    Note:
        This class is a structure to hold examples with included serialization
        and deserialization methods. Therefore, the __init__ method's
        arguments are also the class attributes.

    Attributes:
        guid (str): Unique id for the example. Usually including its subset
            name (e.g. train-5).
        tokens (:obj:`list` of :obj:`str`): The sequence of tokens.
        labels (:obj:`list` of :obj:`str`, optional): The sequence of labels.
            Defaults to None.
    """

    def __init__(self, guid: str, tokens: List[str], labels: List[str] = None):
        """See class."""
        self.guid = guid
        self.tokens = tokens
        self.labels = labels

    def __repr__(self) -> str:
        """String representation of the object."""
        return str(self.to_json_string())

    def to_dict(self) -> Dict[str, ExampleAttType]:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeaturesTC(object):
    """A single set of features of data, probably corresponding to a
    single ``InputExampleTC``.

    In the context of the transformers library, `features` refers to
    a transformer's input e.g. (subword) token ids, attention masks,
    segment ids, label ids, etc.

    Note:
        This class is a structure to hold examples with included serialization
        and deserialization methods. Therefore, the __init__ method's
        arguments are also the class attributes.

    Attributes:
        input_ids (:obj:`list` of :obj:`int`): Indices of input  tokens in the
            vocabulary.
        attention_mask (:obj:`list` of :obj:`int`): Mask to avoid performing
            attention on padding token indices. Mask values selected in
            ``[0, 1]`` usually: ``1`` for tokens that are NOT MASKED,
            ``0`` for MASKED (padded) tokens.
        token_type_ids (:obj:`list` of :obj:`int`): Segment token indices to
            indicate first and second portions of the inputs.
        label_ids (:obj:`list` of :obj:`int`): Label ids corresponding to the
            input token ids.
        token_positions (:obj:`list` of :obj:`int`): positions of original
            tokens.
    """

    def __init__(self,
                 input_ids: List[int],
                 attention_mask: List[int],
                 token_type_ids: List[int],
                 label_ids: List[int],
                 token_positions: List[int]):
        """See class."""
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.token_positions = token_positions

    def __eq__(self, other):
        """Comparison with other objects."""
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        """String representation of the object."""
        return str(self.to_json_string())

    def to_dict(self) -> Dict[str, List[int]]:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class ResultTC(object):
    """A single set of results from inference (token classification).

    Note:
        This class is a structure to hold results with included serialization
        and deserialization methods. Therefore, the __init__ method's arguments
        are also the class attributes.

    Attributes:
        name (str): the token or tokens classified. For NER this contains
            the named entity mention.
        label (str): the label associated with the token or tokens.
            For NER, this is the entity type e.g. ORG.
        token_start (int): the position of the first token for this result.
            For NER, this is the first position of the first token in the
            mention.
        token_end (int): the position of the last token.
        score (float): the score or probability associated with this result.
            For NER, this is the mean of the individual token scores, with
            each token score being the softmax output. A negative number
            can be used to indicate the absence of a score. Defaults to -1.

    Example::

        'If the film In the Mood for Love , ...'
         0  1   2    3  4   5    6   7    8 ...

        ResultTC(
            name="In the Mood for Love",
            label="WORK_OF_ART",
            token_start=3,
            token_end=7,
            score=0.9969802618026733
        )

    """

    def __init__(self,
                 name: str,
                 token_start: int,
                 token_end: int,
                 label: str,
                 score: float = -1.0):
        """Create `ResultTC` and assign classification results to the
        to their respective attributes. See class docstring."""
        self.name = name
        self.label = label
        self.token_start = token_start
        self.token_end = token_end
        self.score = score

    def __repr__(self) -> str:
        """String representation of object."""
        return str(self.to_json_string())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_txt_string(self) -> str:
        """Serializes this instance to a JSON string without formatting."""
        return json.dumps(self.to_dict())


class DataProcessor(object):
    """Base class for data converters for classification data sets.

    Made to support both sequence classification and token classification
    tasks.
    """

    def get_train_examples(self):
        """Gets a collection of ``InputExampleTC`` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of ``InputExampleTC`` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of ``InputExampleTC`` for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file: str, quotechar: str = None) -> List[str]:
        """Reads a tab separated value file.

        Args:
            input_file (str): the path to the file to read.
            quotechar (str, optional): the CSV quotechar. Defaults to None.
        """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_conll_file(cls,
                         input_file: str,
                         label_col: int = -1) -> List[Tuple[str, str]]:
        """Reads a CONLL format text file.

        Args:
            input_file (str): the path to the file to read.
            label_col (int, optional): the index of the CONLL column to read
                as the label. Defaults to -1 which in python indexing means
                the last column. 0 is assumed to be the tokens.
        """
        examples = []
        with open(input_file, encoding="utf-8") as f:
            tokens = []
            labels = []
            for line in f:
                # check for new sequence (doc/sentence)
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        # end of setence, add example
                        examples.append((tokens, labels))
                        tokens = []
                        labels = []
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
                    labels.append(splits[label_col].strip())
                # end of line
            # end of lines
            # if we have any tokens left, this means that there was no
            # final empty line at the end of the file but there was still
            # and example
            if tokens:
                examples.append((tokens, labels))
        # end of with open
        return examples


class CONLLProcessor(DataProcessor):
    r"""Processor for a CONLL style datasets.

    The dataset is assumed to have the following structure::

        data_dir
        |-- train.txt
        |-- dev.txt
        |-- test.txt
        |-- labels.txt

    With labels.txt being an ordered, line-delimited list of labels e.g.::

        O
        B-PER
        I-PER
        ...

    While train.txt, dev.txt, and test.txt are CONLL style formated files e.g::

        -DOCSTART- -X- -X- O

        EU NNP B-NP B-ORG
        rejects VBZ B-VP O
        German JJ B-NP B-MISC

    """

    def __init__(self, data_dir: str, label_col: int = -1):
        """Create CONNLLProcessor.

        Args:
            data_dir (str): directory containing the data files.
            label_col (int, optional): the index of the CONLL column to read
                as the label. Defaults to -1 which in python indexing means
                the last column. 0 is assumed to be the tokens.
        """
        self.data_dir = data_dir
        self.label_col = label_col

        labels_file = os.path.join(data_dir, 'labels.txt')
        self.labels = open(labels_file).read().splitlines()
        # skip any empty lines accidentally there
        self.labels = [lbl.strip() for lbl in self.labels if lbl.strip()]

    def get_train_examples(self) -> List[InputExampleTC]:
        """See base class."""
        input_file = os.path.join(self.data_dir, 'train.txt')
        logger.info(f"LOOKING AT {input_file}")
        return self._create_examples(self._read_conll_file(input_file,
                                                           self.label_col),
                                     "train")

    def get_dev_examples(self) -> List[InputExampleTC]:
        """See base class."""
        input_file = os.path.join(self.data_dir, 'dev.txt')
        logger.info(f"LOOKING AT {input_file}")
        return self._create_examples(self._read_conll_file(input_file,
                                                           self.label_col),
                                     "dev")

    def get_test_examples(self) -> List[InputExampleTC]:
        """See base class."""
        input_file = os.path.join(self.data_dir, 'test.txt')
        logger.info(f"LOOKING AT {input_file}")
        return self._create_examples(self._read_conll_file(input_file,
                                                           self.label_col),
                                     "test")

    def get_labels(self) -> List[str]:
        """See base class."""
        return self.labels

    def _create_examples(self, lines, set_type) -> List[InputExampleTC]:
        """Creates examples for the different subsets (splits)."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            tokens = line[0]
            labels = line[1]
            ex = InputExampleTC(guid=guid, tokens=tokens, labels=labels)
            examples.append(ex)
        return examples


def convert_example_to_features(example: InputExampleTC,
                                label2id: Dict[str, int],
                                tokenizer: Any,
                                max_length: int = 512,
                                ignore_lbl_id: int = -100
                                ) -> Tuple[InputFeaturesTC, List[str]]:
    """Converts a single ``InputExampleTC``  to a ``InputFeaturesTC``.

    Args:
        example (:obj:`InputExampleTC`): an example to convert to featurs.
        label2id (:obj:`dict` key :obj:`str`, value :obj:`int`): a dictionary
            mapping label strings to their respective label ids.
        tokenizer (:obj): the transformer tokenizer object.
        max_length (int): the maximum length of the post-tokenized tokens and
            the respective associated fields in an InputFeaturesTC. Sequences
            longer will be truncated, sequences shorter will be padded.
            This length includes any special tokens that must be added such
            as [CLS] and [SEP] in BERT.
        ignore_lbl_id (int, optional): a value of a label id to be ignored,
            used for subword tokens. This is typically negative.
            Usually, -1 or `torch.nn.CrossEntropy().ignore_index`.

    Returns:
        (tuple): tuple containing:

            features (:obj:`InputFeaturesTC`) containing the data in `example`
                converted into `features`. Given a task-specific
                ``InputExamplesTC`` it should  return a task-specific
                ``InputFeaturesTC`` for that task.
            sw_tokens (:obj:`list` of :obj:`str`) containing a list of
                (probably) subword tokens as tokenized by the `tokenizer`.

    Raises:
        AssertionError: If lengths of the respective InputFeaturesTC will not
            match.
    """

    label_ids = []
    tokens = []
    positions = []
    for pos, (word, label) in enumerate(zip(example.tokens, example.labels)):
        sub_tokens = tokenizer.tokenize(word)
        sub_pos = [pos] * len(sub_tokens)

        tokens.extend(sub_tokens)
        positions.extend(sub_pos)

        # In BERT NER, only the first subword token of a word is considered
        # for token classification the rest should be ignored
        # in (pytorch) we can use CrossEntropyLoss().ignore_index
        sub_labels = [ignore_lbl_id] * (len(sub_tokens) - 1)
        sub_labels = [label2id[label]] + sub_labels
        label_ids.extend(sub_labels)

    # In BERT for sequence pairs, each token gets the segment id of the
    # sequence it belongs to (0 or 1). But for single sequence tasks
    # only 0 is used.
    #  tokens:      [CLS] the dog is hairy . [SEP]
    #  token_type_ids:   0   0   0   0  0     0   0

    # check for empty examples
    if len(tokens) == 0:
        logger.warn(f'Empty example: {example}')

    # truncate
    tokens = tokens[:(max_length - 2)]
    label_ids = label_ids[:(max_length - 2)]
    positions = positions[:(max_length - 2)]

    # convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # add special token ids: [CLS] tokens [SEP]
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    # add the label_ids
    label_ids = [ignore_lbl_id] + label_ids + [ignore_lbl_id]
    # token type id is 0 for single sentence tasks
    token_type_ids = [0] * len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens.
    # Where `real` includes CLS/SEP (checked in the original code)
    attention_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)

    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    label_ids = label_ids + ([ignore_lbl_id] * padding_length)

    n = len(input_ids)
    assert n == max_length, f'Error input length {n} vs {max_length}'
    n = len(attention_mask)
    assert n == max_length, f'Error attention length {n} vs {max_length}'
    n = len(token_type_ids)
    assert n == max_length, f'Error token_type_ids length {n} vs {max_length}'
    n = len(label_ids)
    assert n == max_length, f'Error label_ids length {n} vs {max_length}'

    features = InputFeaturesTC(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               label_ids=label_ids,
                               token_positions=positions)
    return features, tokens


def log_example_features(example: InputExampleTC,
                         features: InputFeaturesTC,
                         tokens: List[str]) -> None:
    """Logs an ``InputExampleTC`` and its conversion to ``InputFeaturesTC``."""
    logger.info("*** Example ***")
    logger.info("guid: %s" % (example.guid))
    logger.info("text: %s" % " ".join(example.tokens))
    logger.info("labels: %s" % " ".join(example.labels))
    logger.info("tokens: %s" % " ".join(tokens))
    logger.info("input_ids: %s" %
                " ".join([str(x) for x in features.input_ids]))
    logger.info("attention_mask: %s" %
                " ".join([str(x) for x in features.attention_mask]))
    logger.info("token_type_ids: %s" %
                " ".join([str(x) for x in features.token_type_ids]))
    logger.info("label_ids: %s" %
                " ".join([str(x) for x in features.label_ids]))
    logger.info("token_positions: %s" %
                " ".join([str(x) for x in features.token_positions]))


def convert_examples_to_features(examples: Sequence[InputExampleTC],
                                 labels: List[str],
                                 tokenizer: Any,
                                 max_length: int = 512,
                                 ignore_lbl_id: int = -100
                                 ) -> List[InputFeaturesTC]:
    """Converts sequence of ``InputExampleTC to list of ``InputFeaturesTC``.

    Args:
        examples (:obj:`list` of :obj:`InputExampleTC`): Sequence of
            ``InputExampleTC`` containing the examples to be converted to
            features.
        tokenizer (:obj): Instance of a transformer tokenizer that will
            tokenize the example tokens and convert them to model specific ids.
        max_length (int): the maximum length of the post-tokenized tokens and
            the respective associated fields in an InputFeaturesTC. Sequences
            longer will be truncated, sequences shorter will be padded.
            This length includes any special tokens that must be added such
            as [CLS] and [SEP] in BERT.
        ignore_lbl_id (int, optional): a value of a label id to be ignored,
            used for subword tokens. This is typically negative.
            Usually, -1 or `torch.nn.CrossEntropy().ignore_index`.

    Returns:
        If the input is a list of ``InputExamplesTC``, will return
        a list of task-specific ``InputFeaturesTC`` which can be fed to the
        model.

    """

    logger.info(f'Using label list {labels}')
    label2id = {label: i for i, label in enumerate(labels)}

    all_features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Converting example %d" % (ex_index))

        feats, tks = convert_example_to_features(example=example,
                                                 label2id=label2id,
                                                 tokenizer=tokenizer,
                                                 max_length=max_length,
                                                 ignore_lbl_id=ignore_lbl_id)
        if ex_index < 5:
            log_example_features(example, feats, tks)
        all_features.append(feats)

    return all_features


def convert_tokens_to_example(tokens: List[str],
                              guid: str = '',
                              labels: List[str] = None) -> InputExampleTC:
    """Creates an ``InputExampleTC`` from a list of tokens.

    Note:
        This function is meant to be used at inference time (i.e. not during
        training or evaluation) where data is expected not to have labels.

    Args:
        tokens (:obj:`list` of :obj:`str`): Sequence of tokens to be converted
            to an ``InputExampleTC``.
        guid (str, optional): a unique identifier for the example. Defaults
            to empty string.
        labels (:obj:`list` of :obj:`str`): a label for each corresponding
            token. Since this function is primarly made for inference, it
            defaults to a list of empty strings that exist for compatability.
            However a different string signifying no label could be used.


    Returns:
        ``InputExampleTC`` containing the tokens passed as input.

    """

    if labels is None:
        labels = [''] * len(tokens)
    return InputExampleTC(guid=guid, tokens=tokens, labels=labels)


def convert_features_to_pytorch_dataset(all_features: List[InputFeaturesTC]
                                        ) -> TensorDataset:
    """Converts a list of features into a pytorch dataset.

    Args:
        all_features (:obj:`list` of :obj:`InputFeatureTC`): the list of
            ``InputFeatureTC`` originating from a list of ``InputExampleTC``
            that will constitute the dataset.

    Returns:
        A pytorch TensorDataset containing the features with the attributes
        of features occupying the following dimensions:

            0 - input (token) ids
            1 - attention mask
            2 - token types (or segment ids)
            3 - label ids

    """

    all_input_ids = torch.tensor([x.input_ids for x in all_features],
                                 dtype=torch.long)
    all_attention_mask = torch.tensor([x.attention_mask for x in all_features],
                                      dtype=torch.long)
    all_token_type_ids = torch.tensor([x.token_type_ids for x in all_features],
                                      dtype=torch.long)
    all_label_ids = torch.tensor([x.label_ids for x in all_features],
                                 dtype=torch.long)
    # Create Tensor dataset
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids)
    return dataset


def convert_features_to_dataset(all_features: List[InputFeaturesTC],
                                dataset_type: str = 'pytorch'
                                ) -> TensorDataset:
    """Converts a list of features into a dataset.

    Args:
        all_features (:obj:`list` of :obj:`InputFeatureTC`): the list of
            ``InputFeatureTC`` originating from a list of ``InputExampleTC``
            that will constitute the dataset.
        dataset_type (str): the type of dataset, curruntly only `pytorch` is
            supported.

    Returns:
        A pytorch TensorDataset.

    Raises:
        ValueError if `dataset_type` is not supported.
    """

    if dataset_type == 'pytorch':
        all_input_ids = torch.tensor([x.input_ids for x in all_features],
                                     dtype=torch.long)
        all_attention_mask = torch.tensor([x.attention_mask
                                           for x in all_features],
                                          dtype=torch.long)
        all_token_type_ids = torch.tensor([x.token_type_ids
                                           for x in all_features],
                                          dtype=torch.long)
        all_label_ids = torch.tensor([x.label_ids
                                      for x in all_features],
                                     dtype=torch.long)
        # Create Tensor dataset
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids)
    else:
        raise ValueError(f'Invalid return dataset type: {dataset_type}')

    return dataset


class DataBunchLoaderTC(object):
    """A class for grouping multiple subsets of a token classficiation dataset.

    The idea is that training and evaluation commonly require a train,
    development (or validation), and test subsets (or splits). Generally all
    of these have the same parameters (e.g. max sequence length) and same
    set of labels, etc. This class is for made for objects to contain all the
    required parameters and options for the 3 subsets of a token classification
    dataset.

    See ``CONLLProcessor`` for the directory structure and data file
    requirements.

    Attributes:
        tokenizer: the (subword) tokenizer of the pretrained transformer model.

    """

    def __init__(self,
                 data_dir: str,
                 slen: int,
                 backend: str = 'pytorch',
                 cached_data_dir: str = None):
        """Creates a DataBunchLoaderTC object.

        Args:
            data_dir (str): path to the directory containing the dataset.
            slen (int): maximum sequence length with longer sequences being
                truncated and shorter sequences being padded.
            backend (str): only `pytorch` is currently supported.
            cached_data_dir (str): path to a directory where cached features
                will be stored or loaded from.

        Raises:
            ValueError if `backend` parameter is invalid.
        """
        self.data_dir = data_dir
        self.slen = slen
        self.backend = backend.lower()
        self.cached_data_dir = cached_data_dir
        self.processor = CONLLProcessor(data_dir)
        self.ignore_lbl_id = -1
        if backend == 'pytorch':
            self.ignore_lbl_id = torch.nn.CrossEntropyLoss().ignore_index

        self._tokenizer = None

    def get_labels(self) -> List[str]:
        """Returns the list of labels."""
        return self.processor.get_labels()

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        """Set the tokenizer."""
        self._tokenizer = tokenizer

    def _path_for_cached(self, subset: str = 'train') -> str:
        """Given a subset name, return the path for its cached data."""
        fname = subset + '.' + str(self.slen) + '.pt'
        fpath = os.path.join(self.cached_data_dir, fname)
        return fpath

    def _load_cached(self, subset: str = 'train') -> List[InputFeaturesTC]:
        """Load a cached subset."""
        if not self.cached_data_dir:
            return None
        fpath = self._path_for_cached(subset)
        if os.path.isfile(fpath):
            logger.info(f'Loading cached {subset} from {fpath}')
            return torch.load(fpath)
        return None

    def _save_cached(self,
                     features: List[InputFeaturesTC],
                     subset: str = 'train') -> bool:
        """Save features to a cache file."""
        if not self.cached_data_dir:
            return False

        if not os.path.exists(self.cached_data_dir):
            os.makedirs(self.cached_data_dir)

        fpath = self._path_for_cached(subset)
        logger.info(f'Saving cached {subset} at {fpath}')
        torch.save(features, fpath)
        return True

    def _get_features(self, subset: str = 'train') -> List[InputFeaturesTC]:
        """Features for a subset from either cache or by generating them.

        If features are not already cached and if cached_data_dir is not None,
        it will cache the features generated.

        Returns:
            List of ``InputFeaturesTC`` for the subset.
        """
        features = self._load_cached(subset)
        if features is not None:
            return features

        if not self._tokenizer:
            raise Exception('Tokenizer not set: call tokenizer(tokenizer)')

        examples = None
        if subset == 'train':
            examples = self.processor.get_train_examples()
        elif subset == 'dev':
            examples = self.processor.get_dev_examples()
        elif subset == 'test':
            examples = self.processor.get_test_examples()
        else:
            raise KeyError(f'No subset: {subset}')

        features = convert_examples_to_features(examples, self.processor,
                                                self._tokenizer,
                                                self.slen,
                                                self.ignore_lbl_id)
        self._save_cached(features, subset)
        return features

    def get_train(self, batch_size: int) -> DataLoader:
        """Returns the train dataloader."""
        features = self._get_features('train')
        # convert features do dataset
        dataset = convert_features_to_dataset(features, self.backend)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler,
                                batch_size=batch_size)
        return dataloader

    def get_dev(self, batch_size: int) -> DataLoader:
        """Returns the dev dataloader."""
        features = self._get_features('dev')
        # convert features do dataset
        dataset = convert_features_to_dataset(features, self.backend)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler,
                                batch_size=batch_size)
        return dataloader

    def get_test(self, batch_size: int = 1) -> DataLoader:
        """Returns the test dataloader."""
        features = self._get_features('test')
        # convert features do dataset
        dataset = convert_features_to_dataset(features, self.backend)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler,
                                batch_size=batch_size)
        return dataloader
