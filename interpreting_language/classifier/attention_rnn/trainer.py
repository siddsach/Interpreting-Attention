import torchtext
from torchtext import data
from torchtext.vocab import Vectors
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.autograd import Variable
from torch.optim import Adam, SGD
from model import VanillaRNN, SelfAttentiveRNN
import time
import glob
import os
from datetime import datetime
import pickle
import argparse

root_path = os.getcwd()
print("ROOT_PATH: {}".format(root_path))

#### DEFAULTS ####
SPLIT = 0.75
DATASET = 'IMDB'
IMDB_PATH = root_path + '/data/imdb/aclImdb'# 'sentence_subjectivity.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
MPQA_PATH = root_path + '/data/mpqa/mpqa_subj_labels.pickle'
VECTOR_CACHE = root_path + '/vectors'
SAVED_VECTORS = True
NUM_EPOCHS = 40
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
LOG_INTERVAL = 20
WORD_VEC_DIM = 200
WORDVEC_SOURCE = ['GloVe']
TUNE_WORDVECS = False
#['GloVe']# charLevel'
MODEL_SAVEPATH = None#'saved_model.pt'
IMDB = True
HIDDEN_SIZE = 300
PRETRAINED = None #root_path + '/trained_models/trained_rnn.pt'
MAX_LENGTH = 100
SAVE_CHECKPOINT = root_path + '/trained_models/classifier/'
ATTN_TYPE = 'keyval'
ATTENTION_DIM = 350 if ATTN_TYPE is not None else None
L2 = 0.001
DROPOUT = 0.5
RNN_DROPOUT = 0.0
MLP_HIDDEN = 512
OPTIMIZER = 'adam'
CLIP = 1
NUM_LAYERS = 1
HIDDEN_SIZE = 300

MAX_DATA_LEN = 1000
if torch.cuda.is_available():
    MAX_DATA_LEN = None

parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
parser.add_argument('--attention', type=str, default=None,
                    help='location of the data corpus')
parser.add_argument('--pretrained', type=str, default = PRETRAINED,
                    help='location, of pretrained init')
parser.add_argument('--checkpoint', type=str, default = None,
                    help='location, of pretrained init')
parser.add_argument('--data',  type=str, default = DATASET,
                    help='location of pretrained init')
parser.add_argument('--num_epochs',  type=int, default = NUM_EPOCHS,
                    help='location of pretrained init')
parser.add_argument('--lr', type=float, default = LEARNING_RATE,
                    help='location of pretrained init')
parser.add_argument('--l2', type=float, default = L2,
                    help='location of pretrained init')
parser.add_argument('--batch_size', type=int, default = BATCH_SIZE,
                    help='location of pretrained init')
parser.add_argument('--model_type', type=str, default = "LSTM",
                    help='location of pretrained init')
parser.add_argument('--num_layers', type=int, default = NUM_LAYERS,
                    help='location of pretrained init')
parser.add_argument('--hidden_size', type=int, default = HIDDEN_SIZE,
                    help='location of pretrained init')
parser.add_argument('--attention_dim', type=int, default = ATTENTION_DIM,
                    help='location of pretrained init')
parser.add_argument('--mlp_hidden', type=int, default = MLP_HIDDEN,
                    help='location of pretrained init')
parser.add_argument('--wordvec_dim', type=int, default = WORD_VEC_DIM,
                    help='location of pretrained init')
parser.add_argument('--wordvec_source', type=list, default = WORDVEC_SOURCE,
                    help='location of pretrained init')
parser.add_argument('--tune_wordvecs', type=list, default = TUNE_WORDVECS,
                    help='location of pretrained init')
parser.add_argument('--max_length', type=int, default = MAX_LENGTH,
                    help='location of pretrained init')
parser.add_argument('--optim', type=str, default = 'adam',
                    help='location of pretrained init')
parser.add_argument('--dropout', type=float, default = DROPOUT,
                    help='location of pretrained init')
parser.add_argument('--rnn_dropout', type=float, default = RNN_DROPOUT,
                    help='location of pretrained init')
parser.add_argument('--max_data_len', type=int, default = MAX_DATA_LEN,
                    help='location of pretrained init')
parser.add_argument('--clip', type=float, default = CLIP,
                    help='location of pretrained init')
parser.add_argument('--savepath', type=str, default = MODEL_SAVEPATH,
                    help='location of pretrained init')
args = parser.parse_args()

def sorter(example):
    return len(example.text)

class TrainClassifier:
    def __init__(
                    self,
                    num_classes = 2,
                    pretrained_modelpath = args.pretrained,
                    checkpoint = args.checkpoint,
                    datapath = args.data,
                    num_epochs = args.num_epochs,
                    lr = args.lr,
                    vector_cache = VECTOR_CACHE,
                    objective = 'nllloss',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    batch_size = args.batch_size,
                    model_type = args.model_type,
                    num_layers = args.num_layers,
                    hidden_size = args.hidden_size,
                    attention_dim = args.hidden_size, #None if not using attention
                    mlp_hidden = args.mlp_hidden,
                    wordvec_dim = args.wordvec_dim,
                    wordvec_source = args.wordvec_source,
                    tune_wordvecs = args.tune_wordvecs,
                    max_length = args.max_length,
                    use_cuda = True,
                    savepath = args.savepath,
                    optim = args.optim,
                    max_data_len = args.max_data_len,
                    dropout = args.dropout,
                    rnn_dropout =  args.rnn_dropout,
                    clip = args.clip,
                    attn_type = args.attention,
                    l2 = args.l2
                ):

        self.savepath = savepath

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA!")
            self.cuda = True
        else:
            print("Not Using CUDA")
            self.cuda = False

        self.lr = lr
        self.datapath = datapath
        if self.datapath == 'MPQA':
            self.filepath = MPQA_PATH
        elif self.datapath == 'IMDB':
            self.trainpath = IMDB_PATH + '/train'
            self.testpath = IMDB_PATH + '/test'

        self.max_data_len = max_data_len

        #TRAINING PARAMETERS
        self.batch_size = batch_size
        self.n_epochs = num_epochs
        self.vector_cache = vector_cache
        self.log_interval = log_interval

        if objective == 'crossentropy':
            self.objective = CrossEntropyLoss()

        elif objective == 'nllloss':
            self.objective = NLLLoss()

        #MODEL ARCHITECTURE SPECS
        self.model_type = model_type
        self.attn_type = attn_type
        self.attention_dim = attention_dim if self.attn_type is not None else None
        self.mlp_hidden = mlp_hidden
        self.num_classes = num_classes

        #HYPERPARAMS
        self.wordvec_source = wordvec_source
        self.wordvec_dim = wordvec_dim
        self.tune_wordvecs = tune_wordvecs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pretrained_modelpath = pretrained_modelpath
        self.checkpoint_path = checkpoint
        self.max_length = max_length
        self.optim = optim
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.l2 = l2
        self.clip = clip

        self.accuracies = torch.zeros(self.n_epochs)

        self.sentence_field = data.Field(
                            sequential = True,
                            use_vocab = True,
                            init_token = '<BOS>',
                            eos_token = '<EOS>',
                            fix_length = self.max_length,
                            include_lengths = True,
                            preprocessing = None, #function to preprocess if needed, already converted to lower, probably need to strip stuff
                            tensor_type = torch.LongTensor,
                            lower = True,
                            tokenize = 'spacy',
                            batch_first = True
                        )

        self.target_field = data.Field(sequential = False, batch_first = True)



    def get_data(self, path, fields, max_len, file_split = True):

        print("Retrieving Data from file: {}...".format(path))

        if self.datapath == 'IMDB':

            examples = []

            for label in ['pos', 'neg']:
                c = 0
                for fname in glob.iglob(os.path.join(path, label, '*.txt')):

                    if max_len is not None:
                        if c > max_len:
                            break

                    with open(fname, 'r') as f:
                        text = f.readline()
                    examples.append(data.Example.fromlist([text, label], fields))

                    c += 1

            out = torchtext.data.Dataset(examples, fields)
            return out

        elif self.datapath == 'MPQA':

            mpqa_data = pickle.load(open(self.filepath, 'rb'))
            mpqa_data[1] = [str(el) for el in mpqa_data[1]]

            n_ex = len(mpqa_data[0])
            boundary = int(SPLIT*n_ex)

            train_data, train_labels = mpqa_data[0][:boundary], mpqa_data[1][:boundary]
            test_data, test_labels = mpqa_data[0][boundary:], mpqa_data[1][boundary:]

            train_examples = []
            test_examples = []

            c = 0
            for text, label in zip(train_data, train_labels):


                train_examples.append(data.Example.fromlist([text, label], fields))
                if max_len is not None:
                    if c > max_len:
                        break
                c += 1

            c = 0
            for text, label in zip(test_data, test_labels):
                test_examples.append(data.Example.fromlist([text, label], fields))
                if max_len is not None:
                    if c > max_len:
                        break
                c += 1

            return data.Dataset(train_examples, fields), data.Dataset(test_examples, fields)





    def load_data(self):
        fields = [('text', self.sentence_field),
                  ('label', self.target_field)]
        if self.datapath == 'IMDB':

            if self.trainpath is not None:
                self.train_data = self.get_data(self.trainpath, fields, max_len = self.max_data_len)

            if self.testpath is not None:
                if self.max_data_len is not None:
                    self.max_data_len = self.max_data_len / 4
                self.test_data = self.get_data(self.testpath, fields, max_len = self.max_data_len)
        elif self.datapath == 'MPQA':

            self.train_data, self.test_data = self.get_data(self.filepath, fields, self.max_data_len)


        else:
            self.sentences = data.TabularDataset(
                            path = self.datapath,
                            format = 'csv',
                            fields = fields
                        )

    def get_vectors(self):
        vecs = []
        print('Loading Vectors From Memory...')

        if len(self.wordvec_source) == 0:
            print('Not using pretrained wordvectors')
            assert self.tune_wordvecs, "You're using random vectors and not tuning them, how do you think that'll pan out?"
        else:
            print('Using these vectors: {}'.format(self.wordvec_source))

        for source in self.wordvec_source:
            if source == 'GloVe':
                print('Getting GloVe Vectors with {} dims'.format(self.wordvec_dim))
                glove = Vectors(name = 'glove.6B.{}d.txt'.format(self.wordvec_dim), cache = self.vector_cache)
                vecs.append(glove)
            if source == 'charLevel':
                print('Getting charLevel Vectors')
                charVec = Vectors(name = 'charNgram.txt', cache = self.vector_cache)
                vecs.append(charVec)
            if source == 'googlenews':
                print('Getting google news vectors')
                google = Vectors(name = 'googlenews.bin', cache = self.vector_cache)
                vecs.append(google)
        print('Building Vocab...')
        if len(vecs) > 0:
            self.sentence_field.build_vocab(self.train_data, vectors = vecs)
            self.target_field.build_vocab(self.train_data)
        else:
            self.sentence_field.build_vocab(self.train_data)
            self.target_field.build_vocab(self.train_data)



    def build_batches(self, dataset):
        print('Getting Batches...')
        if self.cuda:
            iterator_object = data.BucketIterator(dataset,
                                            sort_key = sorter,
                                            batch_size = self.batch_size,
                                            sort = True
                                        )
            iterator_object.repeat = False
        else:
            iterator_object = data.BucketIterator(dataset,
                                            sort_key = sorter,
                                            sort = True,
                                            batch_size = self.batch_size,
                                            device = -1
                                        )
            iterator_object.repeat = False

        print("Created Iterator with {num} batches".format(num = iterator_object))
        return iterator_object

    def get_batches(self):
        if self.train_data is not None:
            self.train_iterator = self.build_batches(self.train_data)
        if self.test_data is not None:
            self.test_iterator = self.build_batches(self.test_data)

    def get_model(self, num_tokens = None):
        if self.checkpoint_path is None:
            print('Building model...')

            self.ntokens = len(self.sentence_field.vocab)

            pretrained_model = None
            if self.pretrained_modelpath is not None:
                pretrained_model = torch.load(self.pretrained_modelpath)
                print('Using Pretrained RNN from path: {}'.format(self.pretrained_modelpath))

            args = {'vocab_size' : self.ntokens,
                'num_classes' : self.num_classes,
                'batch_size' : self.batch_size,
                'cuda' : self.cuda,
                'vectors' : self.sentence_field.vocab.vectors,
                'pretrained_rnn' : pretrained_model,
                'input_size' : self.wordvec_dim,
                'dropout' : self.dropout,
                'rnn_dropout' : self.rnn_dropout,
                'hidden_size' : self.hidden_size,
                'num_layers' : self.num_layers,
                'train_word_vecs' : self.tune_wordvecs
            }


            if self.attention_dim is None:
                self.model = VanillaRNN(**args)
                print('Using Vanilla RNN with following args:\n{}'
                        .format(args))

            else:
                attn_args = {
                    'attention_dim' : self.attention_dim,
                    'mlp_hidden' : self.mlp_hidden,
                    'attn_type' : self.attn_type,
                }
                args += attn_args

                print('Using Attention model with following args:\n{}'
                        .format(args))
                self.model = SelfAttentiveRNN(**args)

                #MAKING MATRIX TO SAVE ATTENTION WEIGHTS
                self.train_attns = torch.zeros(2, len(self.train_data), self.max_length)
                self.eval_attns = torch.zeros(2, len(self.test_data), self.max_length)


            if self.cuda:
                self.model.cuda()
        else:
            print('Loading Model from checkpoint')
            self.model = torch.load(self.checkpoint_path)


    def repackage_hidden(self, h):
        '''Wraps hidden states in new Variables, to detach them from their history.'''
        if type(h) == Variable:
            if self.cuda:
                return Variable(h.data.type(torch.FloatTensor).cuda())
            else:
                return Variable(h.data.type(torch.FloatTensor))
        else:
            return tuple(self.repackage_hidden(v) for v in h)


    def evaluate(self):
        self.model.eval()
        i = 0
        accuracies = torch.zeros(len(self.test_iterator))
        total_loss = 0
        for i, batch in enumerate(self.test_iterator):
            #GETTING TENSORS
            data, targets = batch.text, batch.label.view(-1)
            data, lengths = data[0], data[1]
            targets = targets - 1

            #CONVERTING TO CUDA IF ON NEEDED
            if self.cuda:
                data = data.cuda()
                targets = targets.cuda()
                lengths = lengths.cuda()

            if data.size(0) == self.batch_size:

                #GETTING PREDICTIONS
                output, h, A = self.model(data, lengths = lengths)
                predictions = output.view(-1, self.num_classes)

                accuracies[i] = get_accuracy(predictions, targets)

                if A is not None and False:
                    #SAVING ATTENTION WEIGHTS
                    self.save_attns(i, data, A, "test")

                #CALCULATING LOSS
                loss = self.objective(predictions, targets)
                total_loss += loss.data


        self.eval_accuracy = float(torch.sum(accuracies)) / float(torch.nonzero(accuracies).size(0))
        print('Done Evaluating: Achieved accuracy of {}'
                .format(self.eval_accuracy))

    def train_step(self, optimizer, start_time):

        accuracies = torch.zeros(self.log_interval)
        total_loss = 0

        for i, batch in enumerate(self.train_iterator):
            #CLEARING HISTORY
            optimizer.zero_grad

            #GETTING TENSORS
            data, targets = batch.text, batch.label.view(-1)
            targets = targets - 1 #from zero to one
            data, lengths = data[0], data[1]

            #CONVERTING TO CUDA IF ON NEEDED
            if self.cuda:
                data = data.cuda()
                targets = targets.cuda()
                lengths = lengths.cuda()

            if data.size(0) == self.batch_size:
                #GETTING PREDICTIONS
                output, h, A = self.model(data, lengths = lengths)
                predictions = output.view(-1, self.num_classes)
                accuracies[i % self.log_interval] = get_accuracy(predictions, targets)

                if A is not None and False:
                    #SAVING ATTENTION WEIGHTS
                    self.save_attns(i, data, A, 'train')

                #CALCULATING AND PROPAGATING LOSS
                loss = self.objective(predictions, targets)
                loss.backward()

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                if self.optim in ['adam', 'SGD']:
                    optimizer.step()
                elif self.optim == 'vanilla_grad':
                    parameters = filter(lambda p: p.requires_grad, self.model.parameters())
                    for p in parameters:
                        p.data.add_(-self.lr, p.grad.data)

                total_loss += loss.data


                if i % self.log_interval == 0 and i != 0:
                    current_accuracy = float(torch.sum(accuracies)) / float(torch.nonzero(accuracies).size(0))
                    current_loss = total_loss[0] / self.log_interval
                    total_loss = 0
                    elapsed = time.time() - start_time
                    accuracies = torch.zeros(self.log_interval)
                    print('At time: {elapsed} accuracy is {current_accuracy} and loss is {loss}'\
                            .format(elapsed=elapsed, current_accuracy = current_accuracy, loss = current_loss))

        return optimizer

    def save_attns(self, i, text, attns, fold = 'train'):
        index = self.batch_size * i

        if fold == 'train':
            #SAVE TEXT
            self.train_attns[0, index: index + self.batch_size, :attns.size(1)] = text.data[:, :attns.size(1)]
            #SAVE ATTENTION WEIGHTS
            self.train_attns[1, index: index + self.batch_size, :attns.size(1)] = attns.data

        elif fold == 'test':
            #SAVE TEXT
            self.test_attns[0, index: index + self.batch_size, :attns.size(1)] = text.data[:, :attns.size(1)]
            #SAVE ATTENTION WEIGHTS
            self.test_attns[1, index: index + self.batch_size, :attns.size(1)] = attns.data


    def save_checkpoint(self, optimizer, checkpointpath, name = None):
        state = {
                    'epoch': self.epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_valid_accuracy': self.eval_accuracy,
                    'optimizer': None if optimizer is None else optimizer.state_dict(),
                    'accuracies': self.accuracies
                }
        savepath = checkpointpath + ''.join(str(datetime.now()).split())
        if name is not None:
            savepath = checkpointpath + name

        torch.save(state, savepath)


    def dump_attns(self, attn_path):
        if self.test_attns is not None:
            torch.save(self.test_attns, attn_path)

    def start_train(self):
        print("Building RNN Classifier...")
        self.load_data()
        self.get_vectors()
        self.get_batches()
        self.get_model()
        self.model.train()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = None
        if self.optim == 'adam':
            optimizer = Adam(parameters, lr = self.lr)
        elif self.optim == 'SGD':
            optimizer = SGD(parameters, lr = self.lr, weight_decay = self.l2)

        return optimizer


    def train(self):
        optimizer = self.start_train()

        start_time = time.time()
        print('Begin Training...')

        self.eval_accuracy = 0
        self.best_accuracy = 0
        self.best_model = None

        not_better = 0

        for epoch in range(self.n_epochs):
            print("Completing Train Step at {}th epoch...".format(epoch))
            optimizer = self.train_step(optimizer, start_time)
            print("Evaluating...")
            self.evaluate()
            self.accuracies[epoch] = self.eval_accuracy
            self.epoch = epoch
            if self.eval_accuracy < self.best_accuracy:
                not_better += 1

                if not_better >= 5:
                    if self.optim == 'vanilla_grad':
                        #Annealing
                        self.lr /= 4
                elif not_better >= 10:
                    print('Model not improving. Stopping early with {}'
                           'loss at {} epochs.'.format(self.best_accuracy, self.epoch))
                    break
            else:
                self.best_accuracy = self.eval_accuracy
                self.best_model = self.model
                not_better = 0

        if self.savepath is not None:

            print("Saving Model Parameters and Results...")
            self.save_checkpoint(optimizer, self.savepath)

            print('Finished Training.')

def get_accuracy(predictions, targets):
    preds = torch.max(predictions, dim = 1)[1]
    pct_correct = float(torch.sum(targets == preds)[0].data[0]/predictions.size(0))
    return pct_correct

if __name__ == '__main__':
    trainer = TrainClassifier()
    trainer.train()

