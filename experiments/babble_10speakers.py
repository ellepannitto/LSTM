import os
import sys
import random
import glob

import lstm

random.seed(146)

_model_dir = sys.argv[1]
_corpus_dir = sys.argv[2]
_output_dir = sys.argv[3]

n_iterations = 850
max_len_sent = 100
batches = 10

#500000

for i in range(1, 11):
    n = str(i).zfill(2)

    seed = random.randint(0,1000)

    model_str = _model_dir+"/{}/*.pt".format(n)
#    print(model_str)
    model_path = list(glob.glob(model_str))[0]
#    print(list(glob.glob(model_str)))
#    input()

    corpus_path = _corpus_dir+"/{}/".format(n)
    output_path = _output_dir+"/{}/".format(n)

    os.makedirs(output_path, exist_ok=True)

    lstm.core.babbling.babble(output_path, model_path, 
                              corpus_path, n_iterations, 
                              max_len_sent, batches, seed)