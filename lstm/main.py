import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
from email import parser

import logging.config
import os
from random import choices

from lstm.utils import config_utils as cutils
from lstm.core import network_pipeline as network
from lstm.core import babbling as babbling

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)


def _optimize(args):
    output_dir = args.output_dir
    input_dir = args.input_dir

    init_points = args.init_points
    n_iter = args.number_iterations

    train_batch_size_min = args.train_batch_size_min
    train_batch_size_max = args.train_batch_size_max
    emsize_min = args.embedding_size_min
    emsize_max = args.embedding_size_max
    nhid_min = args.hidden_size_min
    nhid_max = args.hidden_size_max
    learning_rate_min = args.learning_rate_min
    learning_rate_max = args.learning_rate_max
    epochs_min = args.epochs_min
    epochs_max = args.epochs_max
    seq_length_min = args.seq_length_min
    seq_length_max = args.seq_length_max

    eval_batch_size = args.eval_batch_size
    nlayers = args.hidden_layers
    dropout = args.dropout_value
    tied = args.tie_weights
    model_type = args.model_type
    clip = args.clip
    log_interval = args.log_interval

    seed = args.seed
    cuda = args.cuda

    network.optimize(output_dir, input_dir,
                     init_points, n_iter,
                     train_batch_size_min, train_batch_size_max,
                     emsize_min, emsize_max,
                     nhid_min, nhid_max,
                     learning_rate_min, learning_rate_max,
                     epochs_min, epochs_max,
                     seq_length_min, seq_length_max,
                     eval_batch_size, clip, log_interval, dropout, nlayers, tied, model_type, seed, cuda)


def _train(args):
    output_dir = args.output_dir
    input_dir = args.input_dir

    emsize = args.embedding_size
    nhid = args.hidden_size
    nlayers = args.hidden_layers
    dropout = args.dropout_value
    tied = args.tie_weights
    model_type = args.model_type
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    seq_length = args.seq_length
    clip = args.clip
    learning_rate = args.learning_rate
    log_interval = args.log_interval

    seed = args.seed
    cuda = args.cuda

    network.main(output_dir, input_dir, 
                  epochs, train_batch_size, eval_batch_size, seq_length, 
                  clip, learning_rate, log_interval,
                  emsize, nhid, nlayers, dropout, tied, model_type, 
                  seed, cuda)


def _babble(args):
    output_dir = args.output_dir
    model_fpath = args.model_path
    number_of_words = args.words_number
    seed = args.seed

    babbling.babble(output_dir, model_fpath, number_of_words, seed)


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
    parser_train.add_argument('--model-type', type=str, choices=["LSTM", "GRU"],
                              default="RNN", help="model name")
    parser_train.add_argument('--epochs', type=int, default=25,
                              help="number of epochs for training")
    parser_train.add_argument('--train-batch-size', type=int, default=5,
                              help="size of batch size for training")
    parser_train.add_argument('--eval-batch-size', type=int, default=20,
                              help="size of batch size for evaluation")
    parser_train.add_argument('--seq-length', type=int, default=128,
                              help='? DONT REMEMBER')
    parser_train.add_argument('--clip', type=float, default=0.5)
    parser_train.add_argument('--learning-rate', type=float, default=0.8,                # TODO: check values
                              help="learning rate")
    parser_train.add_argument('--log-interval', type=int, default=10,
                              help="??")
    parser_train.add_argument('--seed', type=int, default=42,
                              help="random seed for reproducibility")
    parser_train.add_argument('--cuda', type=bool, default=False,
                              help="cuda flag")
    parser_train.set_defaults(func=_train)


    parser_optimize = subparsers.add_parser("optimize",
                                            description='trains lstm through bayesian optimizer',
                                            help='trains lstm through bayesian optimizer',
                                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser_optimize.add_argument('-o', "--output-dir", default="data/models/",
                                 help="path to output directory")
    parser_optimize.add_argument('-i', '--input-dir', default="data/input_test/",
                                 help="path to directory containing training data")
    parser_optimize.add_argument('--init-points', type=int, default=3,
                                 help="init points for optimization")
    parser_optimize.add_argument('--number-iterations', type=int, default=100,
                                 help="number of iteration for optimization")
    parser_optimize.add_argument('--train-batch-size-min', type=int, default=20,
                                 help="minimum size of batch size for training")
    parser_optimize.add_argument('--train-batch-size-max', type=int, default=40,
                                 help="maximum size of batch size for training")
    parser_optimize.add_argument('--embedding-size-min', type=int, default=350,
                                 help='minimum size of embedding')
    parser_optimize.add_argument('--embedding-size-max', type=int, default=500,
                                 help='maximum size of embedding')
    parser_optimize.add_argument('--hidden-size-min', type=int, default=350,
                                 help='minimum size of hidden layer')
    parser_optimize.add_argument('--hidden-size-max', type=int, default=500,
                                 help='maximum size of hidden layer')
    parser_optimize.add_argument('--learning-rate-min', type=float, default=0.8,                # TODO: check values
                                 help="minimum learning rate")
    parser_optimize.add_argument('--learning-rate-max', type=float, default=1,
                                 help="maximum learning rate")
    parser_optimize.add_argument('--epochs-min', type=int, default=30,
                                 help="minimum number of epochs for training")
    parser_optimize.add_argument('--epochs-max', type=int, default=70,
                                 help="maximum number of epochs for training")
    parser_optimize.add_argument('--seq-length-min', type=int, default=15,
                                 help='? DONT REMEMBER')
    parser_optimize.add_argument('--seq-length-max', type=int, default=50,
                                 help='? DONT REMEMBER')
    parser_optimize.add_argument('--eval-batch-size', type=int, default=20,
                                 help="size of batch size for evaluation")
    parser_optimize.add_argument('--hidden-layers', type=int, default=2,
                                 help='number of hidden layers')
    parser_optimize.add_argument('--dropout-value', type=float, default=0.2,
                                 help='dropout value')
    parser_optimize.add_argument('--tie-weights', type=bool, default=False,
                                 help='tie weights flag')
    parser_optimize.add_argument('--model-type', type=str, choices=["LSTM", "GRU"],
                                 default="LSTM", help="model name")
    parser_optimize.add_argument('--clip', type=float, default=0.25)
    parser_optimize.add_argument('--log-interval', type=int, default=200,
                                 help="??")
    parser_optimize.add_argument('--seed', type=int, default=42,
                                 help="random seed for reproducibility")
    parser_optimize.add_argument('--cuda', type=bool, default=False,
                                 help="cuda flag")
    parser_optimize.set_defaults(func=_optimize)                                        

    parser_babble = subparsers.add_parser("babble",
                                          description='sample babbling from model',
                                          help='sample babbling from model',
                                          formatter_class=ArgumentDefaultsHelpFormatter)
    parser_babble.add_argument('-o', "--output-dir", default="data/models/",
                               help="path to output directory")
    parser_babble.add_argument('-m', "--model-path", required=True,
                               help="filepath to model .pt")
    parser_babble.add_argument('-n', '--words-number', type=int, default=500000,
                               help="number of words to sample")
    parser_babble.add_argument('-s', '--seed', type=int, default=42,
                                help='random seed')
    parser_babble.set_defaults(func=_babble)

    args = root_parser.parse_args()
    if "func" not in args:
        root_parser.print_usage()
        exit()
    args.func(args)
