import logging
import os
import numpy as np

import torch
import torch.nn as nn

from lstm.utils import os_utils as outils
from lstm.utils import corpus_utils as cutils
from lstm.utils import torch_utils as tutils

logger = logging.getLogger(__name__)

def babble(output_path, model_path, corpus_path, n_iterations, max_len_sent, batches):

    for batch_n in range(batches):
        logger.info("BATCH {}".format(batch_n))
        main_sample_parallel(output_path, model_path, corpus_path,
                             n_iterations, max_len_sent, batch_n)


def main_sample_parallel(output_path, model_path, corpus_path,
                         n_iterations, max_len_sent, batch_n):

    

    train_fname, valid_fname, test_fname = outils.get_network_fnames(corpus_path)
    corpus = cutils.Corpus(train_fname, valid_fname, test_fname)

    cuda = True

    model_basename = os.path.basename(model_path).split(".")[0]

    with open(model_path, 'rb') as f:
        model = torch.load(f)
        # model = torch.load(f, map_location=torch.device('cpu'))

    model.eval()

    with torch.no_grad():
        with open(output_path+"babbling-{}-{}.txt".format(model_basename, str(batch_n).zfill(2)), "w") as fout:

            for iteration in range(n_iterations):

                if not iteration % 10:
                    logger.info("PROCESS:{} - iteration {}".format(os.getpid(), iteration))

                hidden = model.init_hidden(1)
                startwith = corpus.sample_first_letter()

                input_data = tutils.batchify([torch.LongTensor([startwith, ])], 1, cuda)

                data = input_data

                s = [startwith]

                number_of_generated_sentences = 0
                number_of_generated_chars_before_newline = 0

                while number_of_generated_sentences < 150:
                    hidden = tutils.repackage_hidden(hidden)
                    model.zero_grad()
                    output, hidden = model(data, hidden)
                    probabilities, top_idx = torch.topk(output[0], k=output.size(2))

                    probabilities = nn.functional.softmax(probabilities[0], dim=0).tolist()
                    probabilities = [el / sum(probabilities) for el in probabilities]
                    choices = top_idx.tolist()

                    choice = np.random.choice(choices[0], p=probabilities)
                    s.append(choice)

                    input_data = nutils.batchify([torch.LongTensor([choice, ])], 1, cuda)
                    data = input_data
                    number_of_generated_chars_before_newline += 1

                    if number_of_generated_chars_before_newline > max_len_sent:
                        s.append(corpus.dictionary.letter2idx["\n"])
                        number_of_generated_sentences += 1
                        number_of_generated_chars_before_newline = 0

                        input_data = nutils.batchify([torch.LongTensor([corpus.dictionary.letter2idx["\n"], ])],
                                                     1, cuda)
                        data = input_data

                    if choice == corpus.dictionary.letter2idx["\n"]:
                        number_of_generated_sentences += 1
                        number_of_generated_chars_before_newline = 0

                sent = [corpus.dictionary.idx2letter[char] for char in s]
                print("".join(sent), file=fout)
                print("\n\n", file=fout)


def parallel_babble(output_dir, model_fpath, corpus_dir, n_iterations, max_sen_len, workers):
    for i in range(workers):
        print("BATCH", i)
        main_sample_parallel(output_dir, model_path, corpus_dir,
                             n_iterations, max_sen_len)