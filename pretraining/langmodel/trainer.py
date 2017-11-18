from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe, CharNGram
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from .model import LangModel
import time
from .nce import NCELoss
import os

current_path = os.getcwd()
project_path = current_path#[:len(current_path)-len('/pretraining/langmodel')]

VECTOR_CACHE = project_path + '/vectors'
SAVED_VECTORS = True
NUM_EPOCHS = 1
BPTT_LENGTH = 35
LEARNING_RATE = 0.5
BATCH_SIZE = 50
LOG_INTERVAL = 5
BPTT_SEQUENCE_LENGTH = 2
WORDVEC_DIM = 300
WORDVEC_SOURCE = ['GloVe']# 'googlenews', 'charLevel']

#TRAIN_PATH = project_path + 'data/gigaword/gigaword_cleaned_small.txt'#'data/wikitext-2/wikitext-2/wiki.train.tokens'

DATASET = 'wiki'
WIKI_PATH = project_path + '/data/wikitext-2/wikitext-2/'
MODEL_SAVE_PATH = project_path + '/trained_models/trained_rnn.pt'
NUM_LAYERS = 1
HIDDEN_SIZE = 4096

def preprocess(x):
    #ENSURE ENCODING IS RIGHT
    try:
        return x.encode('utf-8').decode('utf-8').lower()
    except ValueError:
        print("COULD NOT DECODE FOR EXAMPLE:")
        print(type(x))
        print(x)

def get_freqs(vocab_object):

    num_specials = 4
    vocab_size = len(vocab_object.itos)
    out = torch.zeros(vocab_size)

    for i in range(num_specials, vocab_size):
        out[i] = vocab_object.freqs[vocab_object.itos[i]]

    return out


def build_unigram_noise(freq):
    """build the unigram noise from a list of frequency
    Parameters:
        freq: a tensor of #occurrences of the corresponding index
    Return:
        unigram_noise: a torch.Tensor with size ntokens,
        elements indicate the probability distribution
    """
    total = freq.sum()
    noise = freq / total
    assert abs(noise.sum() - 1) < 0.001
    return noise


class TrainLangModel:
    def __init__(
                    self,
                    data = DATASET,
                    n_epochs = NUM_EPOCHS,
                    seq_len = BPTT_SEQUENCE_LENGTH,
                    lr = LEARNING_RATE,
                    batch_size = BATCH_SIZE,
                    vector_cache = VECTOR_CACHE,
                    objective = 'nce',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    model_type = "LSTM",
                    savepath = MODEL_SAVE_PATH,
                    wordvec_dim = WORDVEC_DIM,
                    wordvec_source = WORDVEC_SOURCE,
                    num_layers = NUM_LAYERS,
                    hidden_size = HIDDEN_SIZE
                ):
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False
        self.savepath = savepath
        self.lr = lr
        self.data = data
        self.batch_size = batch_size
        self.bptt_len = seq_len
        self.n_epochs = n_epochs
        self.vector_cache = vector_cache
        self.objective_function = objective

        self.num_layers = num_layers
        self.hidden_size = hidden_size


        self.log_interval = log_interval
        self.model_type = model_type
        self.wordvec_source = wordvec_source
        self.wordvec_dim = wordvec_dim


    def load_data(self):

        print("Preparing Data Loaders")
        self.sentence_field = data.Field(
                            sequential = True,
                            use_vocab = True,
                            init_token = '<BOS>',
                            eos_token = '<EOS>',
                            preprocessing = data.Pipeline(convert_token = preprocess), #function to preprocess if needed, already converted to lower, probably need to strip stuff
                            tensor_type = torch.LongTensor,
                            lower = True,
                            tokenize = 'spacy',
                        )

        datapath = None
        if self.data == 'wikitext':
            datapath = WIKI_PATH

            paths = [datapath + 'wiki.' + s + '.tokens' for s in ['train', 'valid', 'test']]

        trainpath, validpath, testpath = paths[0], paths[1], paths[2]

        print("Retrieving Train Data from file: {}...".format(trainpath))
        self.train_sentences = datasets.LanguageModelingDataset(trainpath, self.sentence_field, newline_eos = False)
        print('done.')


        if validpath is not None:

            print("Retrieving Valid Data from file: {}...".format(validpath))
            self.valid_sentences = datasets.LanguageModelingDataset(validpath, self.sentence_field, newline_eos = False)
            print('done.')

        if testpath is not None:

            print("Retrieving Test Data from file: {}...".format(testpath))
            self.test_sentences = datasets.LanguageModelingDataset(testpath, self.sentence_field, newline_eos = False)
            print('done.')



    def get_vectors(self):
        vecs = []
        print('Loading Vectors From Memory...')
        for source in self.wordvec_source:
            if source == 'GloVe':
                glove = Vectors(name = 'glove.6B.{}d.txt'.format(self.wordvec_dim), cache = self.vector_cache)
                vecs.append(glove)
            if source == 'charLevel':
                charVec = Vectors(name = 'charNgram.txt',cache = self.vector_cache)
                vecs.append(charVec)
            if source == 'googlenews':
                googlenews = Vectors(name = 'googlenews.txt', cache = self.vector_cache)
                vecs.append(googlenews)
        print('Building Vocab...')
        self.sentence_field.build_vocab(self.train_sentences, vectors = vecs)
        print('Done.')


    def get_iterator(self, dataset):
        print('Getting Batches...')

        if self.cuda:
            iterator = data.BPTTIterator(dataset, sort_key = None, bptt_len = self.bptt_len, batch_size = self.batch_size)
            iterator.repeat = False
        else:
            iterator = data.BPTTIterator(dataset, sort_key = None, bptt_len = self.bptt_len,  batch_size = self.batch_size, device = -1)
            iterator.repeat = False

        print("Done.")
        return iterator

    def get_model(self):
        print('Initializing Model parameters...')
        self.ntokens = self.sentence_field.vocab.__len__()
        if self.model_type == "LSTM":
            print('Constructing LSTM with {} layers and {} hidden size...'.format(self.num_layers, self.hidden_size))
            if self.objective_function == 'crossentropy':
                print('Using Cross Entropy Loss and Softmax activation...')
                self.objective = CrossEntropyLoss()

                self.model = LangModel(vocab_size = self.ntokens,
                                    pretrained_vecs = self.sentence_field.vocab.vectors,
                                    decoder = 'softmax',
                                    num_layers = self.num_layers,
                                    hidden_size = self.hidden_size
                                )

            elif self.objective_function == 'nce':
                print('Using Cross Entropy Loss and Softmax activation...')
                freqs = get_freqs(self.sentence_field.vocab)
                self.noise = build_unigram_noise(freqs)
                self.model = LangModel(vocab_size = self.ntokens,
                                    pretrained_vecs = self.sentence_field.vocab.vectors,
                                    decoder = 'nce',
                                    num_layers = self.num_layers,
                                    hidden_size = self.hidden_size
                                )
                self.objective = NCELoss(self.ntokens, self.model.hidden_size, self.noise, self.cuda)


    def repackage_hidden(self, h):
        #Erasing hidden state history
        if type(h) == Variable:
            return Variable(h.data.type(torch.FloatTensor))
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def train_step(self, optimizer, model, start_time):
        print('Completing Train Step...')
        hidden = self.model.init_hidden(self.batch_size)
        total_loss = 0
        for i, batch in enumerate(self.train_iterator):
            optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            data, targets = batch.text, batch.target.view(-1)
            output, hidden = model(data, hidden)
            if self.objective_function == 'crossentropy':
                output = output.view(-1, self.ntokens)
            else:
                output = output.view(output.size(0) * output.size(1), output.size(2))
            loss = self.objective(output, targets)
            loss.backward()
            total_loss += loss.data
            optimizer.step()
            if i % self.log_interval == 0:
                current_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                total_loss = 0
                print('At time: {elapsed} loss is {current_loss}'.format(elapsed=elapsed, current_loss = current_loss[0]))
        print('Finished Train Step')



    def evaluate(self):
        print('Begin Evaluating...')
        self.model.eval()
        hidden = self.model.init_hidden(self.batch_size)
        total_loss = 0
        self.valid_iterator = self.get_iterator(self.valid_sentences)
        for i, batch in enumerate(self.valid_iterator):
            hidden = self.repackage_hidden(hidden)
            data, targets = batch.text, batch.target.view(-1)
            output, hidden = self.model(data, hidden)
            if self.objective_function == 'crossentropy':
                output = output.view(-1, self.ntokens)
            else:
                output = output.view(output.size(0) * output.size(1), output.size(2))
            loss = self.objective(output, targets)
            total_loss += len(data) * loss.data
        print('Done Evaluating: Achieved loss of {}'.format(total_loss[0]))

    def train(self):
        self.load_data()
        self.get_vectors()
        self.train_iterator = self.get_iterator(self.train_sentences)
        self.get_model()
        self.model.train()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = Adam(parameters)
        start_time = time.time()
        print('Begin Training...')
        for epoch in range(self.n_epochs):
            print('Finished {} epochs...'.format(epoch))
            self.train_step(optimizer, self.model, start_time)
        print('Finished Training.')

    def save_model(self, savepath):
        torch.save(self.model.model.state_dict(), savepath)

if __name__ == '__main__':

    trainer = TrainLangModel()
    trainer.train()
    trainer.evaluate()
    trainer.save_model(trainer.savepath)



'''
print("DATA")
print(type(data.data), type(targets.data))
print(data.data.shape, targets.data.shape)
print("HIDDEN")
print([type(hidden[i].data) for i in range(2)])
print([vec.data.shape for vec in hidden])
print('PREDICTIONS')
print(type(output))
print(output.data.shape)
'''
