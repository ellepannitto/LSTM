import logging

import time
import math
import numpy as np

import torch
import torch.nn as nn

from lstm.core import network_lib as network
from lstm.utils import torch_utils as tutils


logger = logging.getLogger(__name__)

def train(model, train_batch_size, train_data, 
          seq_length, criterion, ntokens, clip, learning_rate, log_interval, epoch, log_file):
   
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    hidden = model.init_hidden(train_batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_length)):
        data, targets = tutils.get_batch(train_data, i, seq_length)
        # truncated BPP
        hidden = tutils.repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            # print("p.data", torch.norm(p.data))
            # print(lr)
            # print("p.grad.data", torch.norm(p.grad.data))
            # print()
            p.data.add_(-learning_rate, p.grad.data)
        # input()

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                   epoch, batch, len(train_data) // seq_length, learning_rate,
                   elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)), file=log_file)
            total_loss = 0
            start_time = time.time()


def evaluate(val_data, model, eval_batch_size, seq_length, ntokens):
    
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, seq_length):
            data, targets = tutils.get_batch(val_data, i, seq_length)

            # > output has size seq_length x batch_size x vocab_size
            output, hidden = model(data, hidden)
            # > output_flat has size num_targets x vocab_size (batches are stacked together)
            # > ! important, otherwise softmax computation (e.g. with F.softmax()) is incorrect
            output_flat = output.view(-1, ntokens)
            # output_candidates_info(output_flat.data, targets.data)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets).item()
            hidden = tutils.repackage_hidden(hidden)

    return total_loss / (len(val_data) - 1)


def sample(model, corpus, cuda, seed):
    
    model.eval()

    with torch.no_grad():
        ret = []
        for _ in range(5):
            hidden = model.init_hidden(1)
            startwith = corpus.sample_first_letter()
            input_data = tutils.batchify([torch.LongTensor([startwith,])], 1, cuda, seed)

            data = input_data

            s = [startwith]

            for seq in range(100):
                hidden = tutils.repackage_hidden(hidden)
                model.zero_grad()
                output, hidden = model(data, hidden)
                probabilities, top_idx = torch.topk(output[0], k=output.size(2))

                probabilities = nn.functional.softmax(probabilities[0]).tolist()
                probabilities = [el/sum(probabilities) for el in probabilities]
                choices = top_idx.tolist()

                choice = np.random.choice(choices[0], p=probabilities) # TODO: add seed

                s.append(choice)

                input_data = tutils.batchify([torch.LongTensor([choice,])], 1, cuda, seed)

                data = input_data

            ret.append(s)

    return corpus.decode(ret)