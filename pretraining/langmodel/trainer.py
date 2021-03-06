from torchtext import data, datasets
from torchtext.vocab import Vocab, Vectors, CharNGram
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from collections import Counter
from .model import LangModel
from time import time
#from nce import NCELoss
import os
import math
from datetime import datetime
import argparse
import sys
import os
import psutil

current_path = os.getcwd()
project_path = current_path#[:len(current_path)-len('/pretraining/langmodel')]

TIME_LIMIT = None#3 * 60 * 60
DATASET = 'gigaword'
WIKI_PATH = project_path + '/data/wikitext-2/wikitext-2/'
PTB_PATH = project_path + '/data/penn/'
GIGA_PATH = project_path + '/data/gigaword/'
MODEL_SAVE_PATH = project_path + '/trained_models/langmodel/'
VECTOR_CACHE = project_path + '/vectors'

NUM_EPOCHS = 1 if not torch.cuda.is_available() else 100
LEARNING_RATE = 5
LOG_INTERVAL = 50
BPTT_SEQUENCE_LENGTH = 35
BATCH_SIZE = 20
WORDVEC_DIM = 300
GLOVE_DIM = 300
WORDVEC_SOURCE = 'gigavec'#GloVe', 'charLevel']
CHARNGRAM_DIM = 100
TUNE_WORDVECS = False
PRETRAINED_WORDVEC = False
CLIP = 0.25
NUM_LAYERS = 3
TIE_WEIGHTS = True
MODEL_TYPE = 'LSTM'
OPTIMIZER = 'vanilla_grad'
DROPOUT = 0.4
RNN_DROPOUT = 0.2
HIDDEN_SIZE = 1024
FEW_BATCHES = 1000 if not torch.cuda.is_available() else None
MAX_VOCAB = 2000000
MIN_FREQ = 5
ANNEAL = 4.0
REINIT_ARGS = [
                    'num_epochs',
                    'seq_len',
                    'clip',
                    'batch_size',
                    'vector_cache',
                    'objective_function',
                    'log_interval',
                    'model_type',
                    'savepath',
                    'wordvec_dim',
                    'glove_dim',
                    'wordvec_source',
                    'tune_wordvecs',
                    'num_layers',
                    'hidden_size',
                    'tie_weights',
                    'optim',
                    'dropout',
                    'rnn_dropout',
                    'few_batches',
                    'anneal',
                    'current_batch'
            ]


class TrainLangModel:
    def __init__(
                    self,
                    num_epochs = NUM_EPOCHS,
                    seq_len = BPTT_SEQUENCE_LENGTH,
                    lr = LEARNING_RATE,
                    clip = CLIP,
                    batch_size = BATCH_SIZE,
                    vector_cache = VECTOR_CACHE,
                    objective_function = 'crossentropy',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    model_type = MODEL_TYPE,
                    savepath = MODEL_SAVE_PATH,
                    wordvec_dim = WORDVEC_DIM,
                    glove_dim = GLOVE_DIM,
                    wordvec_source = WORDVEC_SOURCE,
                    tune_wordvecs = TUNE_WORDVECS,
                    num_layers = NUM_LAYERS,
                    hidden_size = HIDDEN_SIZE,
                    use_cuda = True,
                    tie_weights = TIE_WEIGHTS,
                    optim = OPTIMIZER,
                    dropout = DROPOUT,
                    rnn_dropout = DROPOUT,
                    few_batches = FEW_BATCHES,
                    anneal = ANNEAL,
                    current_batch = 0,
                    charlevel = False,
                    time_limit = None,
                    max_vocab = MAX_VOCAB
                ):

        if torch.cuda.is_available() and use_cuda:
            self.cuda = True
        else:
            self.cuda = False
        self.lr = lr
        self.savepath = savepath

        self.model_type = model_type
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.optim = optim
        self.anneal = anneal
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.few_batches = few_batches
        self.time_limit = time_limit
        self.max_vocab = max_vocab

        if self.time_limit is not None:
            print("AVAILABLE TIME:{}".format(self.time_limit))

        self.num_epochs = num_epochs

        self.num_layers = num_layers

        self.clip = clip



        #HYPERPARAMS
        if charlevel:
            print("BUILDING CHAR LEVEL RNN: GETTING RID OF WORDVEC ATTS")
            self.tokenizer = lambda x: list(x)
            self.glove_dim = None
            self.wordvec_source = None
            self.pretrained_vecs = False
            self.wordvec_dim = wordvec_dim
        else:
            self.tokenizer = 'spacy'
            self.glove_dim = glove_dim
            self.wordvec_source = wordvec_source
            self.pretrained_vecs = self.wordvec_source in ['google', 'glove', 'charlevel', 'gigavec']
            self.wordvec_dim = 0 if self.pretrained_vecs else wordvec_dim

        self.hidden_size = hidden_size

        self.tie_weights = tie_weights


        self.tune_wordvecs = True if not self.pretrained_vecs else tune_wordvecs

        self.objective_function = objective_function

        self.vector_cache = vector_cache

        self.log_interval = log_interval
        self.current_batch = current_batch
        self.current_loss = []
        self.epoch = 0
        self.time = time


    #IF already_read is False, data is the name of a dataset, else it is the dataset itself
    def load_data(self, dataset, more = False, examples = None, already_read = True):
        print("Preparing Data Loaders")
        self.sentence_field = data.Field(
                    sequential = True,
                    use_vocab = True,
                    init_token = '<BOS>',
                    eos_token = '<EOS>',
                    #function to preprocess
                    preprocessing = data.Pipeline(convert_token = preprocess),
                    tensor_type = torch.LongTensor,
                    lower = True,
                    tokenize = 'spacy'
                )
        fields = [('text', self.sentence_field)]

        if not already_read:

            datapath = None
            trainpath, validpath, testpath = None, None, None

            if dataset == 'wikitext':
                datapath = WIKI_PATH

                paths = [datapath + 'wiki.' + s + '.tokens' for s \
                                     in ['train', 'valid', 'test']]

                trainpath, validpath, testpath = paths[0], paths[1], paths[2]

            elif dataset == 'ptb':
                datapath = PTB_PATH
                paths = [datapath + s + '.txt' for s in ['train', 'valid', 'test']]

                trainpath, validpath, testpath = paths[0], paths[1], paths[2]

            elif dataset == 'gigaword':
                datapath = GIGA_PATH
                trainpath = datapath + 'thread5.txt'

            elif dataset == 'gigasmall':
                datapath = GIGA_PATH
                trainpath = datapath + 'gigaword_small_train.txt'
                validpath = datapath + 'gigaword_small_val.txt'
                testpath = datapath + 'gigaword_small_test.txt'

            elif dataset == 'reviews':
                trainpath = 'data/reviews/reviews.txt'

            print("Retrieving Train Data from file: {}...".format(trainpath))
            start = time()
            self.train_sentences = datasets.LanguageModelingDataset(trainpath,\
                    self.sentence_field, newline_eos = False)
            finish = time() - start
            print("Downloaded in {} minutes".format(finish / 60))
            print("Got Train Dataset with {n_tokens} words".format(n_tokens =\
                    len(self.train_sentences.examples[0].text)))


            if validpath is not None:

                print("Retrieving Valid Data from file: {}...".format(validpath))
                self.valid_sentences = datasets.LanguageModelingDataset(validpath,\
                        self.sentence_field, newline_eos = False)
            else:
                self.valid_sentences = None

            if testpath is not None:

                print("Retrieving Test Data from file: {}...".format(testpath))
                self.test_sentences = datasets.LanguageModelingDataset(testpath,\
                        self.sentence_field, newline_eos = False)
            else:
                self.test_sentences = None
        elif more:
	

            if examples is None:
                examples = []

            already_split = False
            if already_split:
                for i, fold in enumerate(dataset):
                    print('Reading fold:{}'.format(i))
                    pid = os.getpid()
                    py = psutil.Process(pid)
                    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
                    print('memory use:', memoryUse)
                    examples.append(data.Example.fromlist([fold], fields))
                self.train_sentences = data.Dataset(examples, fields)
                self.sentence_field.build_vocab(self.train_sentences)
                return self.sentence_field.freqs
                '''
                print("EXAMPLES")
                print(len(examples))
                print([ex.text[:100] for ex in examples])
                print("EXAMPLES END")
                '''
            else:
                examples.append(data.Example.fromlist([dataset], fields))#[i*fold_size:(i+1)*fold_size 

            one = time()
            #print("READ EXAMPLES IN {}".format(one - start))
            self.train_sentences = data.Dataset(examples, fields)

        

    def get_vectors(self, vocab):
        sources = None
        if self.wordvec_source == 'glove':
            sources = ['GloVe']
        elif self.wordvec_source == 'charlevel':
            sources = ['GloVe', 'charLevel']
        elif self.wordvec_source == 'google':
            sources = ['googlenews']
        elif self.wordvec_source == 'gigavec':
            sources = ['gigavec']
        else:
            sources = []

        print('Building Vocab...')

        if isinstance(vocab, Vocab):
            print("Using Pretrained Vocab")
            self.sentence_field.vocab = vocab
            print(len(self.sentence_field.vocab.itos))
        else:
            print('wrong')
            vecs = []
            print('Loading Vectors From Memory...')
            if self.pretrained_vecs:
                print('Using these vectors: ' + str(self.wordvec_source))
                for source in sources:
                    if source == 'GloVe':
                        glove = Vectors(name = 'glove.6B.{}d.txt'
                                .format(self.glove_dim), cache = self.vector_cache)
                        vecs.append(glove)
                        self.wordvec_dim += self.glove_dim
                    if source == 'charLevel':
                        charVec = CharNGram()
                        self.wordvec_dim += 100
                    if source == 'googlenews':
                        googlenews = Vectors(name = 'googlenews.txt',\
                                cache = self.vector_cache)
                        vecs.append(googlenews)
                        self.wordvec_dim += 300
                    if source == 'gigavec':
                        gigavec = Vectors(name = 'gigamodel.vec',\
                                cache = self.vector_cache)
                        vecs.append(gigavec)
                        self.wordvec_dim += 300

            
            if isinstance(vocab, Counter):
                self.sentence_field.vocab = Vocab(vocab, vectors = vecs, max_size = self.max_vocab)
            else:
                self.sentence_field.build_vocab(self.train_sentences, vectors = vecs, \
                        max_size = self.max_vocab, min_freq = MIN_FREQ)
                print('Found {} tokens'.format(len(self.sentence_field.vocab)))


        if self.tie_weights:
            self.hidden_size = self.wordvec_dim

    def get_iterator(self, dataset):

        if self.cuda:
            iterator = data.BPTTIterator(dataset, sort_key = None,\
                    bptt_len = self.seq_len, batch_size = self.batch_size)
            iterator.repeat = False
        else:
            iterator = data.BPTTIterator(dataset, sort_key = None,\
                    bptt_len = self.seq_len,  batch_size = self.batch_size, device = -1)
            iterator.repeat = False

        print("Created Iterator with {num} batches".format(num = len(iterator)))
        return iterator

    def get_model(self, checkpoint = None):
        print('Initializing Model parameters...')
        self.ntokens = len(self.sentence_field.vocab)
        print('Constructing {} with {} layers and {} hidden size...'
                .format(self.model_type, self.num_layers, self.hidden_size))


        model = None

        pretrained_vecs = None
        if self.pretrained_vecs and checkpoint is None:
            pretrained_vecs = self.sentence_field.vocab.vectors
            print("PRETRAINED VECS")
            print(pretrained_vecs.shape)

        if checkpoint is not None:
            self.wordvec_dim = checkpoint["embed.weight"].size(1)


        if self.objective_function == 'crossentropy':
            print('Using Cross Entropy Loss ...')
            self.objective = CrossEntropyLoss()


            model = LangModel(vocab_size = self.ntokens,
                                pretrained_vecs = pretrained_vecs,
                                checkpoint = checkpoint,
                                decoder = 'softmax',
                                num_layers = self.num_layers,
                                hidden_size = self.hidden_size,
                                rnn_dropout = self.rnn_dropout,
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
                                rnn_dropout = self.rnn_dropout,
                                linear_dropout = self.dropout,
                                input_size = self.wordvec_dim,
                                tune_wordvecs = self.tune_wordvecs
                            )
            #self.objective = NCELoss(self.ntokens, self.model.hidden_size,\
                    #self.noise, self.cuda)

        if self.cuda:
            model = model.cuda()

        return model

    def prepare_data(self, data, already_read = False, vocab = None):
        self.load_data(data, already_read = already_read, more = True, examples = None)
        self.get_vectors(vocab)
        print('train examples')
        print(len(self.train_sentences.examples))
        print(len(self.train_sentences.examples[0].text))
        self.train_iterator = self.get_iterator(self.train_sentences)

    def init_model(self, checkpoint_params = None, best_params = None):
        self.model = self.get_model(checkpoint_params)
        self.model.train()

        self.best_loss = 10000000000
        self.best_model = self.get_model(best_params)

        optimizer = None
        scheduler = None
        if self.optim == 'adam':

            parameters = filter(lambda p: p.requires_grad, \
                    self.model.parameters())
            optimizer = Adam(parameters, lr = self.lr, betas = (0, 0.999),\
                    eps = 10**-9)

            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',\
                    factor = 0.1)

        return optimizer, scheduler


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

            if i >= self.current_batch:
                elapsed = time() - start_time
                self.current_batch = i

                if self.time_limit is not None:
                    if elapsed > self.time_limit:
                        print('REACHED TIME LIMIT!')
                        self.save_checkpoint()
                        break

                hidden = self.repackage_hidden(hidden)
                data, targets = batch.text, batch.target.view(-1)

                if self.cuda:
                    data = data.cuda()
                    targets = targets.cuda()

                output, hidden = model(data, hidden)

                if self.objective_function == 'crossentropy':
                    output = output.view(-1, self.ntokens)
                else:
                    output = output.view(output.size(0) * output.size(1), \
                            output.size(2))

                loss = self.objective(output, targets)
                loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(),\
                            self.clip)

                total_loss += loss.data[0]

                if self.optim == 'adam':
                    optimizer.step()
                    optimizer.zero_grad()

                elif self.optim == 'vanilla_grad':
                    parameters = filter(lambda p: p.requires_grad,\
                            self.model.parameters())
                    for p in parameters:
                        p.data.add_(-self.lr, p.grad.data)

                if self.few_batches is not None:
                    if i >= self.few_batches:
                        break

                if ((i + 1) % self.log_interval) == 0:
                    self.current_loss.append(total_loss / self.log_interval)
                    total_loss = 0
                    print('At time: {time} and batch: {i}, loss is {loss}'
                            ' and perplexity is {ppl}'.format(i=i+1, time=elapsed,
                            loss=self.current_loss[-1], ppl=math.exp(self.current_loss[-1])))
        print('Finished Train Step')
        self.current_batch = 0

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
                output = output.view(output.size(0) * output.size(1), \
                        output.size(2))

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



    def train(self, optimizer, scheduler):
        start_time = time()
        print('Begin Training...')

        not_better = 0
        self.epoch = 0
        for epoch in range(self.num_epochs):
            print('finished {} epochs...'.format(epoch))
            elapsed = time() - start_time
            if self.time_limit is not None:
                if elapsed > self.time_limit:
                    break
            self.epoch += 1
            optimizer = self.train_step(optimizer, self.model, start_time)
            this_perplexity = self.evaluate()
            self.epoch = epoch

            if this_perplexity > self.best_loss:
                not_better += 1

                print("Annealing...")
                self.lr /= self.anneal


                if self.optim == 'adam':
                    scheduler.step(this_perplexity)

                if not_better >= 5:
                    print('Model not improving. Stopping early with {} loss'
                           'at {} epochs.'.format(self.best_loss, self.epoch))
                    break

            else:
                self.best_loss = this_perplexity
                self.best_model = self.model
                not_better = 0

        self.best_accuracy = - self.best_loss

        print('Finished Training.')


    def save_checkpoint(self, name = None):

        print("Saving Model Parameters and Results...")

        args = {name: self.__dict__[name] for name in REINIT_ARGS}
        state = {
                    'args': args,
                    'epoch': self.epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_model': self.best_model.state_dict(),
                    'train_loss': self.current_loss,
                    'best_loss': self.best_loss,
                    'vocab': self.sentence_field.vocab
                }
        savepath = None
        print(self.savepath)
        if name is not None:
            savepath = self.savepath + name
        else:
            savepath = self.savepath + 'model.pt'

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

    parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
    parser.add_argument('--checkpoint', type=str, default = None,
                        help='location, of pretrained init')
    parser.add_argument('--data',  type=str, default = DATASET,
                        help='location of pretrained init')
    parser.add_argument('--num_epochs',  type=int, default = NUM_EPOCHS,
                        help='location of pretrained init')
    parser.add_argument('--lr', type=float, default = LEARNING_RATE,
                        help='location of pretrained init')
    parser.add_argument('--batch_size', type=int, default = BATCH_SIZE,
                        help='location of pretrained init')
    parser.add_argument('--model_type', type=str, default = "LSTM",
                        help='location of pretrained init')
    parser.add_argument('--num_layers', type=int, default = NUM_LAYERS,
                        help='location of pretrained init')
    parser.add_argument('--hidden_size', type=int, default = HIDDEN_SIZE,
                        help='location of pretrained init')
    parser.add_argument('--glove_dim', type=int, default = GLOVE_DIM,
                        help='location of pretrained init')
    parser.add_argument('--wordvec_dim', type=int, default = WORDVEC_DIM,
                        help='location of pretrained init')
    parser.add_argument('--wordvec_source', type=str, default = WORDVEC_SOURCE,
                        help='location of pretrained init')
    parser.add_argument('--tune_wordvecs', type=list, default = TUNE_WORDVECS,
                        help='location of pretrained init')
    parser.add_argument('--dropout', type=float, default = DROPOUT,
                        help='location of pretrained init')
    parser.add_argument('--rnn_dropout', type=float, default = RNN_DROPOUT,
                        help='location of pretrained init')
    parser.add_argument('--clip', type=float, default = CLIP,
                        help='location of pretrained init')
    parser.add_argument('--savepath', type=str, default = MODEL_SAVE_PATH,
                        help='location of pretrained init')
    parser.add_argument('--charlevel', type=str, default = MODEL_SAVE_PATH,
                        help='location of pretrained init')
    args = parser.parse_args()

    if args.checkpoint is None:
        trainer = TrainLangModel(
                            num_epochs = args.num_epochs,
                            lr = args.lr,
                            batch_size = args.batch_size,
                            model_type = args.model_type,
                            num_layers = args.num_layers,
                            hidden_size = args.hidden_size,
                            glove_dim = args.glove_dim,
                            wordvec_source = args.wordvec_source,
                            wordvec_dim = args.wordvec_dim,
                            tune_wordvecs = args.tune_wordvecs,
                            use_cuda = True,
                            savepath = args.savepath,
                            dropout = args.dropout,
                            rnn_dropout =  args.rnn_dropout,
                            clip = args.clip
                        )
        trainer.prepare_data(args.data)
        optimizer, scheduler = trainer.init_model()
        trainer.train(optimizer, scheduler)
        trainer.save_checkpoint(trainer.data + '/model.pt')
    else:
        current = torch.load(current_path + '/trained_models/langmodel/{}/training/'.format(args.data) + args.checkpoint)
        trainer = TrainLangModel(**current['args'])
        optimizer, scheduler = trainer.start_train(
                vocab = current['vocab'],
                checkpoint_params = current['state_dict'], \
                best_params = current['best_model']
            )
        trainer.epoch = current['epoch']
        trainer.current_loss = current['train_loss']
        trainer.best_loss = current['best_loss']
        trainer.train(optimizer, scheduler)






'''
fields = [('text', self.sentence_field)]

num_char = len(dataset)
print('Reading in data with {} chars...'.format(num_char))
num_folds = num_char // 2**27 + 1
print("{} folds".format(num_folds))
start = time()
examples = []
print('DATA LENGTH: {}'.format(len(dataset)))

if num_folds == 0:
examples = [data.Example.fromlist([dataset], fields)]
else:
fold_size = math.ceil(num_char/num_folds)
print("fold size:{}".format(fold_size))
for i in range(num_folds):
    print('Reading fold:{}'.format(i))
    fold = dataset[0:fold_size]
    examples.append(data.Example.fromlist([fold], fields))
    dataset = dataset[fold_size:]


one = time()
print("READ EXAMPLES IN {}".format(one - start))
self.train_sentences = data.Dataset(examples, fields)
print("LOADED DATASET IN {}".format(one - time()))
print("Parsed dataset with size of :{}".format(sys.getsizeof(self.train_sentences)))
print('Done.')
'''
