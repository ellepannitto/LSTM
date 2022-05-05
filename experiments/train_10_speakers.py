import sys
import random
import lstm

random.seed(146)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

# | iter | targ train_batch_size | emsize | nhid | learning_rate | epochs | seq_length |
# | 95 | 1.227 | 23.491 | 383.248 | 480.480 | 0.965 | 38.488 | 20.063 |

# PARAMETERS
epochs = 38
train_batch_size = 23
eval_batch_size = 20
seq_length =  20
clip = 0.25
learning_rate = 0.96
log_interval = 200
emsize = 383
nhid = 480
nlayers = 2
dropout = 0.2
tied = False
model_type = "LSTM"
cuda = True


for i in range(1, 11):
    seed = random.randint(0,1000)
    folder_n = str(i).zfill(2)

    print("READING FOLDER N.{}".format(folder_n))

    input_dir = input_dir+"/{}".format(folder_n)
    output_dir = output_dir+"/{}".format(folder_n)

    lstm.core.network_pipeline.main(output_dir, input_dir, 
         epochs, train_batch_size, eval_batch_size, seq_length, 
         clip, learning_rate, log_interval,
         emsize, nhid, nlayers, dropout, tied, model_type,
         seed, cuda)