"""
Define common constants that are used globally.

Also define a configuration object whose parameters can be set via the command line or loaded from an existing
json file. Here you can add more configuration parameters that should be exposed via the command line. In the code,
you can access them via `config.your_parameter`. All parameters are automatically saved to disk in JSON format.

"""
import argparse
import json
import os
import pprint
import torch


class Constants(object):
    """
    This is a singleton.
    """
    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.DTYPE = torch.float32
            self.CIL_DATA_DIR = "../data"

            # These data paths are for training purpose
            self.MASS_DATA_DIR = "C:\Spring 2022\CIL\mass_data"
            self.DEEPGLOBE_DATA_DIR = "C:\Spring 2022\CIL\deepglobe_data"

            self.EXPERIMENT_DIR = "../experiment"

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration(object):
    """Configuration parameters exposed via the commandline."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser()

        # General.
        parser.add_argument('--data_workers', type=int, default=4, help='Number of parallel threads for data loading.')
        # parser.add_argument('--print_every', type=int, default=200, help='Print stats to console every so many iters.')

        parser.add_argument('--eval_every', type=int, default=1, help='Evaluate validation set every so many BATCHES.')

        parser.add_argument('--tag', default='', help='A custom tag for this experiment.')
        parser.add_argument('--seed', type=int, default=None, help='Random number generator seed.')

        parser.add_argument('--pre_train', type=bool, default=None,
                            help='If pre-train, use Massachusetts Roads Dataset; else use CIL Roads Data')
        parser.add_argument('--pretrain_id', type=int, default=None,
                            help='If using pre-trained model, then specify previous experiment id')


        # Learning configurations.
        parser.add_argument('--lr', type=float, default=0.000008, help='Learning rate.')
        parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs.')
        parser.add_argument('--early_stop_threshold', type=int, default=4, help='Early stopping criterion.')

        parser.add_argument('--bs_cil_train', type=int, default=8, help='Batch size for CIL training set.')
        parser.add_argument('--bs_mass_train', type=int, default=3, help='Batch size for MASS training set.')
        parser.add_argument('--bs_dg_train', type=int, default=4, help='Batch size for DeepGlobe training set.')

        parser.add_argument('--bs_ext_train', type=int, default=120, help='Batch size for EXTERNAL training set.')

        parser.add_argument('--bs_eval', type=int, default=1, help='Batch size for valid/test set.')
        parser.add_argument('--n_val', type=int, default=20, help='Number of img/seg for Validation set')

        config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)
