# coding=utf-8
# Copyright (c) 2019 Luis Rei
# MIT License
""" Configuration base class and utilities."""


import os
import copy
import json
import logging


logger = logging.getLogger(__name__)


class ConfigTC():
    """Token Classification Config class.

    Attributes:
        labels (:obj:`list` of :obj:`str`): labels used in task.
        max_seq_length (int): maximum sequence length.
        task_format (str): task format (e.g. `BIO`, `simple`)

    Note:
        These attributes are required.
        Additional attributes passed to constructor will also be stored.
    """

    FNAME = 'tc.json'

    def __init__(self, **kwargs):
        """See class."""
        # required options
        self.labels = kwargs.pop('labels')
        self.max_seq_length = kwargs.pop('max_seq_length')
        self.task_format = kwargs.pop('task_format')

        # optional, extra fields
        self.__dict__.update(kwargs)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory"""
        assert os.path.isdir(save_directory)
        fpath = os.path.join(save_directory, ConfigTC.FNAME)

        logger.info(f'Saving token classification configuration to {fpath}')

        self.to_json_file(fpath)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Load configuration from file in directory.

        Args:
            model_path (str): directory where the model is saved.
            kwargs (dict, optional): key/value pairs with which to update the
                configuration object after loading.

        """

        fpath = os.path.join(model_path, ConfigTC.FNAME)
        config = cls.from_json_file(fpath)
        config.__dict__.update(kwargs)

        logger.info("Token classification config %s", str(config))

        return config

    @classmethod
    def from_dict(cls, d):
        """Constructs a `ConfigTC` from a Python dictionary of parameters."""
        config = cls(**d)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ConfigTC` from a json file of parameters."""
        json_str = open(json_file, "r", encoding='utf-8').read()
        d = json.loads(json_str)
        return cls.from_dict(d)

    def __eq__(self, other):
        """Comparison with other objects."""
        return self.__dict__ == other.__dict__

    def __repr__(self):
        """String representation of the object."""
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())
