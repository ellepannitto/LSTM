import string
import torch
import numpy as np
import unicodedata
import collections


def unicodeToAscii(s, dictionary):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in dictionary.all_letters
    )


def PlainLinearReader(filepath, dictionary):
    with open(filepath) as fin:
        for line in fin:
            line = unicodeToAscii(line.strip(), dictionary)
            if len(line):
                yield line


def load(dictionary, filename):

    distribution_first_letters = collections.defaultdict(int)
    tot = 0

    ids_list = []
    for sentence in PlainLinearReader(filename, dictionary):
        distribution_first_letters[sentence[0]]+=1
        tot+=1
        ids = torch.LongTensor([dictionary.letter2idx[c] for c in sentence+"\n"])
        ids_list.append(ids)

    distribution_first_letters = {v:c/tot for v, c in distribution_first_letters.items()}

    return ids_list, distribution_first_letters


class Dictionary(object):
    def __init__(self):
        self.letter2idx = {}
        self.idx2letter = []

        self.all_letters = string.ascii_letters + string.digits + " .,;'-\n"
        self.n_letters = len(self.all_letters)

        self.letter2idx = {l: i for i, l in enumerate(self.all_letters)}
        self.idx2letter = [l for l in self.all_letters]

    def __len__(self):
        return self.n_letters


class Corpus(object):
    def __init__(self, filename_train, filename_valid, filename_test):
        self.dictionary = Dictionary()
        self.train, self.distribution_first_letters = load(self.dictionary, filename_train)
        self.valid, _ = load(self.dictionary, filename_valid)
        self.test, _ = load(self.dictionary, filename_test)

    def sample_first_letter(self):
        elements = list(self.distribution_first_letters.keys())
        # TODO: set seed manually
        el_idx = np.random.choice(len(elements), p=list(self.distribution_first_letters.values()))
        return self.dictionary.letter2idx[elements[el_idx]]

    def decode(self, ret):
        s = ""
        for sentence in ret:
            for char in sentence:
                s += self.dictionary.idx2letter[char]
            s+="\n\n"
        return s