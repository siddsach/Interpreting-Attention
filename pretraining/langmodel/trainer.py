from torchtext import data, datasets
from torchtext.vocab import Vectors
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from .model import LangModel
import time
from .nce import NCELoss
import os
import math
from datetime import datetime

current_path = os.getcwd()
project_path = current_path#[:len(current_path)-len('/pretraining/langmodel')]

DATASET = 'ptb'
WIKI_PATH = project_path + '/data/wikitext-2/wikitext-2/'
PTB_PATH = project_path + '/data/penn/'
GIGA_PATH = project_path + '/data/gigaword/'
MODEL_SAVE_PATH = project_path + '/trained_models/langmodel/'
VECTOR_CACHE = project_path + '/vectors'

#TRAIN_PATH = project_path + 'data/gigaword/gigaword_cleaned_small.txt'#'data/wikitext-2/wikitext-2/wiki.train.tokens'

NUM_EPOCHS = 100
LEARNING_RATE = 20
LOG_INTERVAL = 50
BPTT_SEQUENCE_LENGTH = 35
BATCH_SIZE = 20
WORDVEC_DIM = 200
WORDVEC_SOURCE = ['GloVe']
CHARNGRAM_DIM = 100
TUNE_WORDVECS = False
PRETRAINED_WORDVEC = False
CLIP = 0.25
NUM_LAYERS = 2
TIE_WEIGHTS = True
MODEL_TYPE = 'LSTM'
OPTIMIZER = 'vanilla_grad'
DROPOUT = 0.2
HIDDEN_SIZE = 4096
FEW_BATCHES = 50 if not torch.cuda.is_available() else None
MAX_VOCAB = None
MIN_FREQ = 5

class TrainLangModel:
    def __init__(
                    self,
                    data = DATASET,
                    num_epochs = NUM_EPOCHS,
                    seq_len = BPTT_SEQUENCE_LENGTH,
                    lr = LEARNING_RATE,
                    clip = CLIP,
                    batch_size = BATCH_SIZE,
                    vector_cache = VECTOR_CACHE,
                    objective = 'crossentropy',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    model_type = MODEL_TYPE,
                    savepath = MODEL_SAVE_PATH,
                    glove_dim = WORDVEC_DIM,
                    wordvec_source = WORDVEC_SOURCE,
                    tune_wordvecs = TUNE_WORDVECS,
                    num_layers = NUM_LAYERS,
                    hidden_size = HIDDEN_SIZE,
                    use_cuda = True,
                    tie_weights = TIE_WEIGHTS,
                    optim = OPTIMIZER,
                    dropout = DROPOUT,
                    few_batches = FEW_BATCHES,
                    pretrained_wordvecs = PRETRAINED_WORDVEC
                ):
        if torch.cuda.is_available() and use_cuda:
            self.cuda = True
        else:
            self.cuda = False
        self.lr = lr
        self.data = data
        self.savepath = savepath + "/"
        print(self.savepath)

        self.model_type = model_type
        self.batch_size = batch_size
        self.bptt_len = seq_len
        self.optim = optim
        self.dropout = dropout
        self.few_batches = few_batches

        self.n_epochs = num_epochs

        self.num_layers = num_layers


        self.clip = clip

        self.wordvec_source = wordvec_source
        self.glove_dim = glove_dim
        self.wordvec_dim = 0
        self.pretrained_wordvecs = pretrained_wordvecs

        for src in self.wordvec_source:
            if src == 'GloVe':
                self.wordvec_dim += self.glove_dim
            if src == 'charLevel':
                self.wordvec_dim += CHARNGRAM_DIM
            if src == 'googlenews':
                pass

        self.tie_weights = tie_weights

        if self.tie_weights:
            self.hidden_size = self.wordvec_dim
        else:
            self.hidden_size = hidden_size

        self.tune_wordvecs = tune_wordvecs

        self.objective_function = objective

        self.vector_cache = vector_cache

        self.log_interval = log_interval


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
                            tokenize = 'spacy'
                        )

        datapath = None
        trainpath, validpath, testpath = None, None, None

        if self.data == 'wikitext':
            datapath = WIKI_PATH

            paths = [datapath + 'wiki.' + s + '.tokens' for s in ['train', 'valid', 'test']]

            trainpath, validpath, testpath = paths[0], paths[1], paths[2]

        elif self.data == 'ptb':
            datapath = PTB_PATH
            paths = [datapath + s + '.txt' for s in ['train', 'valid', 'test']]

            trainpath, validpath, testpath = paths[0], paths[1], paths[2]

        elif self.data == 'gigaword':
            datapath = GIGA_PATH
            trainpath = datapath + 'gigaword_train.txt'
            validpath = datapath + 'gigaword_val.txt'
            testpath = datapath + 'gigaword_test.txt'

        print("Retrieving Train Data from file: {}...".format(trainpath))
        self.train_sentences = datasets.LanguageModelingDataset(trainpath, self.sentence_field, newline_eos = False)
        print("Got Train Dataset with {n_tokens} words".format(n_tokens=len(self.train_sentences.examples[0].text)))


        if validpath is not None:

            print("Retrieving Valid Data from file: {}...".format(validpath))
            self.valid_sentences = datasets.LanguageModelingDataset(validpath, self.sentence_field, newline_eos = False)

        if testpath is not None:

            print("Retrieving Test Data from file: {}...".format(testpath))
            self.test_sentences = datasets.LanguageModelingDataset(testpath, self.sentence_field, newline_eos = False)



    def get_vectors(self):
        vecs = []
        print('Loading Vectors From Memory...')
        if self.pretrained_wordvecs:
            print('Using these vectors: ' + str(self.wordvec_source))
            for source in self.wordvec_source:
                if source == 'GloVe':
                    glove = Vectors(name = 'glove.6B.{}d.txt'.format(self.glove_dim), cache = self.vector_cache)
                    vecs.append(glove)
                if source == 'charLevel':
                    charVec = Vectors(name = 'charNgram.txt',cache = self.vector_cache)
                    vecs.append(charVec)
                if source == 'googlenews':
                    googlenews = Vectors(name = 'googlenews.txt', cache = self.vector_cache)
                    vecs.append(googlenews)
        print('Building Vocab...')
        self.sentence_field.build_vocab(self.train_sentences, vectors = vecs, max_size = MAX_VOCAB, min_freq = MIN_FREQ)
        print('Found {} tokens'.format(len(self.sentence_field.vocab)))


    def get_iterator(self, dataset):
        print('Getting Batches...')

        if self.cuda:
            iterator = data.BPTTIterator(dataset, sort_key = None, bptt_len = self.bptt_len, batch_size = self.batch_size)
            iterator.repeat = False
        else:
            iterator = data.BPTTIterator(dataset, sort_key = None, bptt_len = self.bptt_len,  batch_size = self.batch_size, device = -1)
            iterator.repeat = False

        print("Created Iterator with {num} batches".format(num = len(iterator)))
        return iterator

    def get_model(self):
        print('Initializing Model parameters...')
        self.ntokens = len(self.sentence_field.vocab)
        print('Constructing {} with {} layers and {} hidden size...'.format(self.model_type, self.num_layers, self.hidden_size))

        pretrained_vecs = None
        if self.pretrained_wordvecs:
            pretrained_vecs = self.sentence_field.vocab.vectors

        if self.objective_function == 'crossentropy':
            print('Using Cross Entropy Loss ...')
            self.objective = CrossEntropyLoss()


            self.model = LangModel(vocab_size = self.ntokens,
                                pretrained_vecs = pretrained_vecs,
                                decoder = 'softmax',
                                num_layers = self.num_layers,
                                hidden_size = self.hidden_size,
                                rnn_dropout = self.dropout,
                                linear_dropout = self.dropout,
                                input_size = self.wordvec_dim
                            )

        elif self.objective_function == 'nce':
            print('Using Noise Contrastive Estimation...')
            freqs = get_freqs(self.sentence_field.vocab)
            self.noise = build_unigram_noise(freqs)
            self.model = LangModel(vocab_size = self.ntokens,
                                pretrained_vecs = pretrained_vecs,
                                tie_weights = self.tie_weights,
                                decoder = 'nce',
                                num_layers = self.num_layers,
                                hidden_size = self.hidden_size,
                                rnn_dropout = self.dropout,
                                linear_dropout = self.dropout,
                                input_size = self.wordvec_dim,
                                tune_wordvecs = self.tune_wordvecs
                            )
            self.objective = NCELoss(self.ntokens, self.model.hidden_size, self.noise, self.cuda)

        if self.cuda:
            self.model.cuda()


    def repackage_hidden(self, h):
        #Erasing hidden state history
        if type(h) == Variable:
            if self.cuda:
                return Variable(h.data.type(torch.FloatTensor).cuda())
            else:
                return Variable(h.data.type(torch.FloatTensor))
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def train_step(self, optimizer, model, start_time):
        print('Completing Train Step...')
        hidden = self.model.init_hidden(self.batch_size)
        total_loss = 0
        for i, batch in enumerate(self.train_iterator):
            hidden = self.repackage_hidden(hidden)
            data, targets = batch.text, batch.target.view(-1)

            if self.cuda:
                data = data.cuda()
                targets = targets.cuda()

            output, hidden = model(data, hidden)

            if self.objective_function == 'crossentropy':
                output = output.view(-1, self.ntokens)
            else:
                output = output.view(output.size(0) * output.size(1), output.size(2))

            loss = self.objective(output, targets)
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

            total_loss += loss.data[0]

            if self.optim == 'adam':
                optimizer.step()
                optimizer.zero_grad()

            elif self.optim == 'vanilla_grad':
                parameters = filter(lambda p: p.requires_grad, self.model.parameters())
                for p in parameters:
                    p.data.add_(-self.lr, p.grad.data)

            if self.few_batches is not None:
                if i >= self.few_batches:
                    break

            if ((i + 1) % self.log_interval) == 0:
                current_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                total_loss = 0
                print('At time: {elapsed} and batch: {i}, loss is {current_loss}'
                        ' and perplexity is {ppl}'.format(i=i+1, elapsed=elapsed,
                        current_loss = current_loss, ppl = math.exp(current_loss)))
        print('Finished Train Step')

        return optimizer



    def evaluate(self):
        print('Begin Evaluating...')
        self.model.eval()
        hidden = self.model.init_hidden(self.batch_size)
        total_loss = 0
        self.valid_iterator = self.get_iterator(self.valid_sentences)
        for i, batch in enumerate(self.valid_iterator):
            hidden = self.repackage_hidden(hidden)
            data, targets = batch.text, batch.target.view(-1)

            if self.cuda:
                data = data.cuda()
                targets = targets.cuda()

            output, hidden = self.model(data, hidden)

            if self.objective_function == 'crossentropy':
                output = output.view(-1, self.ntokens)
            else:
                output = output.view(output.size(0) * output.size(1), output.size(2))

            loss = self.objective(output, targets)
            total_loss += loss.data

            if self.few_batches is not None:
                if i >= self.few_batches:
                    break

        avg_loss = total_loss[0] / i
        perplexity = math.exp(avg_loss)
        print('Done Evaluating: Achieved loss of {} and perplexity of {}'
                .format(avg_loss, perplexity))
        return perplexity


    def start_train(self):
        self.load_data()
        self.get_vectors()
        self.train_iterator = self.get_iterator(self.train_sentences)
        self.get_model()
        self.model.train()

        optimizer = None
        scheduler = None
        if self.optim == 'adam':
            parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = Adam(parameters, lr = self.lr, betas = (0, 0.999), eps = 10**-9)

            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1)

        return optimizer, scheduler


    def train(self):
        optimizer, scheduler = self.start_train()

        start_time = time.time()
        print('Begin Training...')

        not_better = 0
        self.best_eval_perplexity = 10000
        self.best_model = None

        for epoch in range(self.n_epochs):
            print('Finished {} epochs...'.format(epoch))
            optimizer = self.train_step(optimizer, self.model, start_time)
            this_perplexity = self.evaluate()
            self.epoch = epoch
            if this_perplexity > self.best_eval_perplexity:
                not_better += 1
                self.lr /= 4.0

                if self.optim == 'adam':
                    scheduler.step(this_perplexity)

                if not_better >= 10:
                    print('Model not improving. Stopping early with {}'
                           'loss at {} epochs.'.format(self.best_eval_perplexity, self.epoch))
                    break

            else:
                self.best_eval_perplexity = this_perplexity
                self.best_model = self.model

        print('Finished Training.')


    def save_checkpoint(self, name = None):
        print("Saving Model Parameters and Results...")
        state = {
                    'epoch': self.epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_valid_loss': self.best_eval_perplexity,
                }
        savepath = self.savepath + ''.join(str(datetime.now()).split())
        print(self.savepath)
        if name is not None:
            savepath = self.savepath + name

        torch.save(state, savepath)

# HELPER FUNCTIONS
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



if __name__ == '__main__':

    trainer = TrainLangModel()
    trainer.train()
    trainer.save_checkpoint()



