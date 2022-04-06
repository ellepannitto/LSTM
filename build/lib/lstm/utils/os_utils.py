import os

def get_network_fnames(path):

    train_fname = path+"/train.txt"
    valid_fname = path+"/valid.txt"
    test_fname = path+"/test.txt"

    if not os.path.isfile(train_fname):
        exit("Training file not found")
    if not os.path.isfile(valid_fname):
        exit("Validation file not found")
    if not os.path.isfile(test_fname):
        exit("Test file not found")

    return train_fname, valid_fname, test_fname