import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
from email import parser
import logging.config
import os
from random import choices

from parso import parse

from lstm.utils import config_utils as cutils
from lstm.core import network_pipeline as network

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)


def _train(args):
    output_dir = args.output_dir
    input_dir = args.input_dir

    emsize = args.embedding_size
    nhid = args.hidden_size
    nlayers = args.hidden_layers
    dropout = args.dropout_value
    tied = args.tie_weights
    model_type = args.model_type

    seed = args.seed
    cuda = args.cuda

    network.train(output_dir, input_dir, emsize, nhid, nlayers, dropout, tied, model_type, seed, cuda)


def main():
    root_parser = argparse.ArgumentParser(prog='lstm', formatter_class=RawTextHelpFormatter)
    subparsers = root_parser.add_subparsers(title="actions", dest="actions")

    parser_train = subparsers.add_parser('train',
                                         description='trains lstm',
                                         help='trains lstm',
                                         formatter_class=ArgumentDefaultsHelpFormatter)
    parser_train.add_argument('-o', "--output-dir", default="data/models/",
                              help="path to output directory")
    parser_train.add_argument('-i', '--input-dir', default="data/input_test/",
                              help="path to directory containing training data")
    parser_train.add_argument('--embedding-size', type=int, default=128,
                              help='size of embedding')
    parser_train.add_argument('--hidden-size', type=int, default=128,
                              help='size of hidden layer')
    parser_train.add_argument('--hidden-layers', type=int, default=2,
                              help='number of hidden layers')
    parser_train.add_argument('--dropout-value', type=float, default=0.2,
                              help='dropout value')
    parser_train.add_argument('--tie-weights', type=bool, default=False,
                              help='tie weights flag')
    parser_train.add_argument('--model-type', type=choices, choices=["LSTM", "GRU"],
                              default="RNN", help="model name")
    parser_train.add_argument('--seed', type=int, default=42,
                              help="random seed for reproducibility")
    parser_train.add_argument('--cuda', type=bool, default=False,
                              help="cuda flag")
    parser_train.set_defaults(func=_train)

    args = root_parser.parse_args()
    if "func" not in args:
        root_parser.print_usage()
        exit()
    args.func(args)
