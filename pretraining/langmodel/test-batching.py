from torchtext import data
import glob
import torch
import os
batch_size = 64

path = os.getcwd() + '/data/imdb/aclImdb'#

def sorter(example):
    return len(example.text)

sentence_field = data.Field(
                    sequential = True,
                    use_vocab = True,
                    init_token = '<BOS>',
                    eos_token = '<EOS>',
                    fix_length = 100,
                    include_lengths = True,
                    preprocessing = None, #function to preprocess if needed, already converted to lower, probably need to strip stuff
                    tensor_type = torch.LongTensor,
                    lower = True,
                    tokenize = 'spacy',
                    batch_first = True
                )

target_field = data.Field(sequential = False, batch_first = True)


print("Retrieving Data from file: {}...".format(path))

fields = [('text', sentence_field), ('label', target_field)]
examples = []

for label in ['pos', 'neg']:
    c = 0
    for fname in glob.iglob(os.path.join(path, label, '*.txt')):

        with open(fname, 'r') as f:
            text = f.readline()
        examples.append(data.Example.fromlist([text, label], fields))

        c += 1

dataset = data.Dataset(examples, fields)
iterator_object = data.Iterator(dataset,
                            sort_key = sorter,
                            sort = True,
                            batch_size = batch_size,
                            device = -1
                        )

for i, batch in enumerate(iterator_object):
    print(batch.text.data.data.shape)
