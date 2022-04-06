import logging

import torch
import torch.nn as nn

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

import time
import math

from lstm.utils import os_utils as outils
from lstm.utils import corpus_utils as cutils
from lstm.utils import model_utils as mutils
from lstm.utils import torch_utils as tutils

from lstm.core import network_lib as network

logger = logging.getLogger(__name__)

opt_logger = JSONLogger(path="./optimizer_logger.json")


def main(output_dir, input_dir, 
         epochs, train_batch_size, eval_batch_size, seq_length, 
         clip, learning_rate, log_interval,
         emsize, nhid, nlayers, dropout, tied, model_type,
         seed, cuda):

    # parameters:
    # data_folder, batch_size, cuda, model_type, emsize, nhid, nlayers, dropout, tied, learning_rate, epochs,
    # save_path, seq_length, clip, log_interval, seed, eval_batch_size=20

    run_id = "batch{}-em{}-nhid{}-nlayers{}-do{}-lr{}-epo{}-bptt{}".format(train_batch_size, emsize, nhid, nlayers,
                                                                           int(dropout*100), int(learning_rate*100),
                                                                           epochs, seq_length)
    # file names
    output_log = "{}/model-{}.log".format(output_dir, run_id)
    output_babbling = "{}/model-{}.babbling".format(output_dir, run_id)
    output_model = "{}/model-{}.pt".format(output_dir, run_id)

    # file handlers
    output_log_h = open(output_log, "w", buffering=1)
    output_babbling_h = open(output_babbling, "w", buffering=1)


    print('-' * 89, file=output_log_h)
    print('PARAMETERS', file=output_log_h)
    print("| batch size {} | emsize {} | nhid {} | nlayers {} | dropout {:.3f} | lr {:.3f} "
          "| epochs {} | bptt {} |".format(train_batch_size, emsize, nhid, nlayers, dropout, learning_rate,
                                           epochs, seq_length), file=output_log_h)
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
    ntokens = len(corpus.dictionary)

    print("( %.2f )" % (time.time() - start), file=output_log_h)
    print("Vocab size %d", ntokens, file=output_log_h)

    ###############################################################################
    # Build the model
    ###############################################################################

    criterion = nn.CrossEntropyLoss()

    print("Building the model", file=output_log_h)

    model = mutils.RNNModel(ntokens, emsize, nhid, nlayers, dropout, tied, model_type)

    if cuda:
        model.cuda()

    ###############################################################################
    # Training code
    ###############################################################################

    best_val_loss = None

    try:
        last_update = 1
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            print("Batchying..", file=output_log_h)

            # TODO: check if seed is needed here
            train_data = tutils.batchify(corpus.train, train_batch_size, cuda, seed)
            # print("Corpus train size", corpus.train.size())
            val_data = tutils.batchify(corpus.valid, eval_batch_size, cuda, seed)

            network.train(model, train_batch_size, train_data, 
                          seq_length, criterion, ntokens, clip, learning_rate, log_interval, epoch, output_log_h)
            # s = sample()
            # logging.info(s)

            val_loss = network.evaluate(val_data, model, eval_batch_size, seq_length, ntokens)
            print('-' * 89, file=output_log_h)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                    val_loss, math.exp(val_loss)), file=output_log_h)
            print('-' * 89, file=output_log_h)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                last_update = epoch
                print("## Saving model at epoch {}".format(epoch), file=output_log_h)
                with open(output_model, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                learning_rate /= 4.0

            if epoch - last_update > 5:
                print("# EXITING FROM TRAINING AT EPOCH {}".format(epoch), file=output_log_h)
                break

            babbling = network.sample(model, corpus, cuda, seed)
            print("** EPOCH {} **".format(epoch), file=output_babbling_h)
            print(babbling, file=output_babbling_h)

    except KeyboardInterrupt:
        print('-' * 89, file=output_log_h)
        print('Exiting from training early', file=output_log_h)


    # Load the best saved model.
    with open(output_model, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_data = tutils.batchify(corpus.test, eval_batch_size, cuda, seed)
    test_loss = network.evaluate(test_data, model, eval_batch_size, seq_length, ntokens)
    
    print('=' * 89, file=output_log_h)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)),
          file=output_log_h)
    print('=' * 89, file=output_log_h)

    return -test_loss


def optimize(output_dir, input_dir,
             init_points, n_iter,
             train_batch_size_min, train_batch_size_max,
             emsize_min, emsize_max,
             nhid_min, nhid_max,
             learning_rate_min, learning_rate_max,
             epochs_min, epochs_max,
             seq_length_min, seq_length_max,
             eval_batch_size, clip, log_interval, dropout, nlayers, tied, model_type, seed, cuda):

    def func_to_optimize(train_batch_size, emsize, nhid, learning_rate, epochs, seq_length):

        train_batch_size = int(train_batch_size)
        emsize = int(emsize)
        nhid = int(nhid)
        epochs = int(epochs)
        seq_length = int(seq_length)

        return main(output_dir, input_dir,
                    epochs, train_batch_size, eval_batch_size, seq_length,
                    clip, learning_rate, log_interval,
                    emsize, nhid, nlayers, dropout, tied, model_type,
                    seed, cuda)

#                                     batch_size=batch_size,
 #                                    cuda=True,
  #                                   model_type='LSTM',
   #                                  emsize=emsize,
    #                                 nhid=nhid,
     #                                nlayers=nlayers,
      #                               dropout=dropout,
       #                              tied=False,
        #                             learning_rate=learning_rate,
         #                            epochs=epochs,
          #                           save_path=output_path+"/",
           #                          seq_length=seq_length,
            #                         clip=0.25,
             #                        log_interval=200,
              #                       seed=seed,
               #                      eval_batch_size=20
                #                     )

#    func_to_optimize(batch_size=100, emsize=400, nhid=400, nlayers=2, dropout=0.5, learning_rate=0.5, epochs=100, seq_length=40)

    # # Bounded region of parameter space
    # pbounds = {'batch_size': (20, 129), #discrete
    #            'emsize': (40, 401),  #discrete
    #            'nhid': (40, 401), #discrete
    #            'nlayers': (2, 4), #discrete
    #             'dropout': (0, 0.5),
    #            'learning_rate': (0.001, 1),
    #            'epochs': (50, 150), #discrete
    #            'seq_length': (10, 101), #discrete
    #            }

#    pbounds = {'train_batch_size': (20, 40), #discrete
#               'emsize': (350, 500),  #discrete
#               'nhid': (350, 500), #discrete
#               'nlayers': (3, 4), #discrete       # 2
#                'dropout': (0, 0.2),              # 0.2
#               'learning_rate': (0.8, 1),         
#               'epochs': (30, 70), #discrete
#               'seq_length': (15, 50), #discrete
#               }

    pbounds = {'train_batch_size': (train_batch_size_min, train_batch_size_max), #discrete
               'emsize': (emsize_min, emsize_max),  #discrete
               'nhid': (nhid_min, nhid_max), #discrete
               'learning_rate': (learning_rate_min, learning_rate_max),         
               'epochs': (epochs_min, epochs_max), #discrete
               'seq_length': (seq_length_min, seq_length_max), #discrete
               }


    optimizer = BayesianOptimization(
        f=func_to_optimize,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.subscribe(Events.OPTIMIZATION_STEP, opt_logger)

    optimizer.maximize(
        init_points=init_points, # 3
        n_iter=n_iter, # 100
    )

    with open(output_dir+"optimization_results.txt", "w") as fout:
        print("| iter | target | " + " | ".join(el for el in pbounds) + " |", file=fout )
        for i, res in enumerate(optimizer.res):
            print("| {} | {:.3f} | ".format(i, -res["target"]) + " | ".join("{:.3f}".format(res["params"][el]) for el in pbounds) + " |", file=fout)

    print(optimizer.max)