import logging

import torch
import torch.nn as nn

import time

from lstm.utils import os_utils as outils
from lstm.utils import corpus_utils as cutils

logger = logging.getLogger(__name__)

def train(output_dir, input_dir, seed, cuda):

    # parameters:
    # data_folder, batch_size, cuda, model_type, emsize, nhid, nlayers, dropout, tied, learning_rate, epochs,
    # save_path, seq_length, clip, log_interval, seed, eval_batch_size=20

#    run_id = "batch{}-em{}-nhid{}-nlayers{}-do{}-lr{}-epo{}-bptt{}".format(batch_size, emsize, nhid, nlayers,
#                                                                           int(dropout*100), int(learning_rate*100),
#                                                                           epochs, seq_length)
    run_id = "prova"
    # file names
    output_log = "{}/model-{}.log".format(output_dir, run_id)
    output_babbling = "{}/model-{}.babbling".format(output_dir, run_id)
    output_model = "{}/model-{}.pt".format(output_dir, run_id)

    # file handlers
    output_log_h = open(output_log, "w", buffering=1)
    output_babbling_h = open(output_babbling, "w", buffering=1)


    print('-' * 89, file=output_log_h)
    print('PARAMETERS', file=output_log_h)
#    print("| batch size {} | emsize {} | nhid {} | nlayers {} | dropout {:.3f} | lr {:.3f} "
#          "| epochs {} | bptt {} |".format(batch_size, emsize, nhid, nlayers, dropout, learning_rate,
#                                           epochs, seq_length), file=output_log_h)
    print('-' * 89, file=output_log_h)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)

    ###############################################################################
    # Load data
    ###############################################################################

    start = time.time()
    train_fname, valid_fname, test_fname = outils.get_network_fnames(input_dir)
    corpus = cutils.Corpus(train_fname, valid_fname, test_fname)

    print("( %.2f )" % (time.time() - start), file=output_log_h)
    print("Vocab size %d", len(corpus.dictionary), file=output_log_h)