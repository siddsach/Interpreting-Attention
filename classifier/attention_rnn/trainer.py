import torchtext
from torchtext import data
from torchtext.vocab import Vectors
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.autograd import Variable
from torch.optim import Adam, SGD
from .model import VanillaRNN, SelfAttentiveRNN
from time import time
import glob
import os
from datetime import datetime
import pickle
import argparse

root_path = os.getcwd()
print("ROOT_PATH: {}".format(root_path))

#### DEFAULTS ####
SPLIT = 0.9
DATASET = 'IMDB'
# 'sentence_subjectivity.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
IMDB_PATH = root_path + '/data/imdb/aclImdb'
MPQA_PATH = root_path + '/data/mpqa/subj_clf_labels.pickle'
VECTOR_CACHE = root_path + '/vectors'
SAVED_VECTORS = True
NUM_EPOCHS = 60
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
LOG_INTERVAL = 20
WORDVEC_DIM = 300
GLOVE_DIM = WORDVEC_DIM
WORDVEC_SOURCE = 'glove'
TUNE_WORDVECS = 'False'
#['GloVe']# charLevel'
MODEL_SAVEPATH = None#'saved_model.pt'
IMDB = True
HIDDEN_SIZE = 300
PRETRAINED =  None#root_path + '/trained_models/langmodel/ptb/model.pt'
MAX_LENGTH = 100
SAVE_CHECKPOINT = None#root_path + '/trained_models/classifier/'
MODEL_TYPE = 'LSTM'
ATTN_TYPE = 'MLP'# ['keyval', 'mlp']
ATTENTION_DIM = 350 if ATTN_TYPE is not None else None
TUNE_ATTN = "True"
L2 = 0.001
DROPOUT = 0.5
RNN_DROPOUT = 0.0
OPTIMIZER = 'adam'
CLIP = 1
NUM_LAYERS = 1
HIDDEN_SIZE = 300

MAX_DATA_LEN = None
if torch.cuda.is_available():
    MAX_DATA_LEN = None
def sorter(example):
    return len(example.text)

torch.manual_seed(10)

class TrainClassifier:
    def __init__(
                    self,
                    num_classes = 2,
                    pretrained = PRETRAINED,
                    checkpoint = SAVE_CHECKPOINT,
                    data = DATASET,
                    num_epochs = NUM_EPOCHS,
                    lr = LEARNING_RATE,
                    vector_cache = VECTOR_CACHE,
                    objective = 'nllloss',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    batch_size = BATCH_SIZE,
                    model_type = MODEL_TYPE,
                    num_layers = NUM_LAYERS,
                    hidden_size = HIDDEN_SIZE,
                    attention_dim = ATTENTION_DIM, #None if not using attention
                    tune_attn = TUNE_ATTN,
                    wordvec_dim = WORDVEC_DIM,
                    glove_dim = GLOVE_DIM,
                    wordvec_source = WORDVEC_SOURCE,
                    tune_wordvecs = TUNE_WORDVECS,
                    max_length = MAX_LENGTH,
                    use_cuda = True,
                    savepath = MODEL_SAVEPATH,
                    optim = OPTIMIZER,
                    max_data_len = MAX_DATA_LEN,
                    dropout = DROPOUT,
                    rnn_dropout =  RNN_DROPOUT,
                    clip = CLIP,
                    attn_type = ATTN_TYPE,
                    l2 = L2,
                    fix_pretrained = None,
                    weight_saving = True
                ):
        self.savepath = savepath

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA!")
            self.cuda = True
        else:
            print("Not Using CUDA")
            self.cuda = False

        self.lr = lr
        self.datapath = data
        if self.datapath == 'MPQA':
            self.filepath = MPQA_PATH
        elif self.datapath == 'IMDB':
            self.trainpath = IMDB_PATH + '/train'
            self.testpath = IMDB_PATH + '/test'
        elif self.datapath == 'amazon':
            self.filepath = 'data/reviews/review_data.csv'

        self.max_data_len = max_data_len

        #TRAINING PARAMETERS
        self.batch_size = batch_size
        self.n_epochs = num_epochs
        self.vector_cache = vector_cache
        self.log_interval = log_interval
        self.weight_saving = weight_saving

        if objective == 'crossentropy':
            self.objective = CrossEntropyLoss()

        elif objective == 'nllloss':
            self.objective = NLLLoss()

        #MODEL ARCHITECTURE SPECS
        self.model_type = model_type
        print(attn_type)
        self.attn_type = attn_type
        self.attention_dim = None if self.attn_type is None else attention_dim
        self.tune_attn = tune_attn
        self.num_classes = num_classes

        #HYPERPARAMS
        if wordvec_source == 'glove':
            self.wordvec_source = ['GloVe']
        elif wordvec_source == 'charlevel':
            self.wordvec_source = ['GloVe', 'charLevel']
        elif wordvec_source == 'google':
            self.wordvec_source = ['googlenews']
        elif wordvec_source == 'gigavec':
            self.wordvec_source = ['gigavec']
        else:
            self.wordvec_source = []

        self.glove_dim = glove_dim
        if len(self.wordvec_source) == 0:
            self.wordvec_dim = wordvec_dim
        else:
            self.wordvec_dim = 0

        self.tune_wordvecs = tune_wordvecs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pretrained = pretrained
        self.fix_pretrained = fix_pretrained

        self.checkpoint_path = checkpoint
        self.max_length = max_length
        self.optim = optim
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.l2 = l2
        self.clip = clip

        self.accuracies = torch.zeros(self.n_epochs)


    def get_data(self, path, fields, max_len, file_split = True):


        if self.datapath == 'IMDB':
            start = time()
            print("Retrieving Data from file: {}...".format(path))
            examples = []

            for label in ['pos', 'neg']:
                c = 0
                for fname in glob.iglob(os.path.join(path, label, '*.txt')):

                    if max_len is not None:
                        if c > max_len:
                            break

                    with open(fname, 'r') as f:
                        text = f.readline()
                    examples.append(data.Example.fromlist([text,label], fields))

                    c += 1

            out = torchtext.data.Dataset(examples, fields)
            print("Got {} examples in {} seconds".format(len(out), time() - start))
            return out

        elif self.datapath == 'MPQA':

            print("Retrieving Data from file: {}...".format(self.filepath))

            mpqa_data = pickle.load(open(self.filepath, 'rb'))
            mpqa_data[1] = [str(el) for el in mpqa_data[1]]

            n_ex = len(mpqa_data[0])
            boundary = int(SPLIT*n_ex)

            train_data = mpqa_data[0][:boundary]
            train_labels = mpqa_data[1][:boundary]
            test_data = mpqa_data[0][boundary:]
            test_labels = mpqa_data[1][boundary:]

            train_examples = []
            test_examples = []

            c = 0
            for text, label in zip(train_data, train_labels):

                train_examples.append(data.Example.fromlist([text,label],fields))
                if max_len is not None:
                    if c > max_len:
                        break
                c += 1

            c = 0
            for text, label in zip(test_data, test_labels):
                test_examples.append(data.Example.fromlist([text,label],fields))
                if max_len is not None:
                    if c > max_len:
                        break
                c += 1

            return data.Dataset(train_examples, fields),data.Dataset(test_examples,fields)
        elif self.datapath == 'amazon':
            print("Retrieving Data from file: {}...".format(path))
            return data.TabularDataset(path, 'CSV', fields, skip_header = True), None






    def load_data(self):
        self.sentence_field = data.Field(
                            sequential = True,
                            use_vocab = True,
                            init_token = '<BOS>',
                            eos_token = '<EOS>',
                            fix_length = self.max_length,
                            include_lengths = True,
                            #converted to lower, probably need to strip stuff
                            preprocessing = None,
                            tensor_type = torch.LongTensor,
                            lower = True,
                            tokenize = 'spacy',
                            batch_first = True
                        )

        self.target_field = data.Field(sequential = False, batch_first = True)

        fields = [('text', self.sentence_field),
                  ('label', self.target_field)]
        if self.datapath == 'IMDB':

            if self.trainpath is not None:
                self.train_data = self.get_data(self.trainpath, fields, max_len = self.max_data_len)

            if self.testpath is not None:
                if self.max_data_len is not None:
                    self.max_data_len = self.max_data_len / 4
                self.test_data = self.get_data(self.testpath, fields, max_len = self.max_data_len)
            else:
                self.test_data = None

        elif self.datapath == 'MPQA':

            self.train_data, self.test_data = self.get_data(self.filepath, fields, self.max_data_len)
        elif self.datapath == 'amazon':
            self.train_data, self.test_data = self.get_data(self.filepath, fields, self.max_data_len)

        else:
            self.sentences = data.TabularDataset(
                            path = self.datapath,
                            format = 'csv',
                            fields = fields
                        )

    def get_vectors(self, pretrained_vocab = None):
        if pretrained_vocab is None:
            vecs = []
            print('Loading Vectors From Memory...')

            if len(self.wordvec_source) == 0:
                print('Not using pretrained wordvectors')
                assert self.tune_wordvecs, "You're using random vectors and not tuning them, how do you think that'll pan out?"
            else:
                print('Using these vectors: {}'.format(self.wordvec_source))

            for source in self.wordvec_source:
                if source == 'GloVe':
                    print('Getting GloVe Vectors with {} dims'.format(self.glove_dim))
                    glove = Vectors(name = 'glove.6B.{}d.txt'.format(self.glove_dim), cache = self.vector_cache)
                    vecs.append(glove)
                    self.wordvec_dim += self.glove_dim
                if source == 'charLevel':
                    self.wordvec_dim += 100
                    print('Getting charLevel Vectors')
                    charVec = Vectors(name = 'charNgram.txt', cache = self.vector_cache)
                    vecs.append(charVec)
                if source == 'googlenews':
                    print('Getting google news vectors')
                    self.wordvec_dim += 300
                    google = Vectors(name = 'googlenews.txt', cache = self.vector_cache)
                    vecs.append(google)
                if source == 'gigavec':
                    gigavec = Vectors(name = 'gigamodel.vec',\
                            cache = self.vector_cache)
                    vecs.append(gigavec)
                    self.wordvec_dim += 300

            print('Building Vocab...')
            print(vecs)
            if len(vecs) > 0:
                self.sentence_field.build_vocab(self.train_data, vectors = vecs)
            else:
                self.sentence_field.build_vocab(self.train_data)

        else:
            print('Loading Pretrained Vocab...')
            self.sentence_field.vocab = pretrained_vocab

        self.target_field.build_vocab(self.train_data)



    def build_batches(self, dataset):
        print('Getting Batches...')
        if self.cuda:
            iterator_object = data.Iterator(dataset,
                                            sort_key = sorter,
                                            batch_size = self.batch_size,
                                            sort = True
                                        )
            iterator_object.repeat = False
        else:
            iterator_object = data.Iterator(dataset,
                                            sort_key = sorter,
                                            sort = True,
                                            batch_size = self.batch_size,
                                            device = -1
                                        )
            iterator_object.repeat = False

        print("Created Iterator with {num} batches".format(num = len(iterator_object)))
        return iterator_object

    def get_batches(self):
        if self.train_data is not None:
            self.train_iterator = self.build_batches(self.train_data)
        if self.test_data is not None:
            self.test_iterator = self.build_batches(self.test_data)

    def get_model(self, pretrained_weights, pretrained_args, num_tokens = None):
        if self.checkpoint_path is None:
            print('Building model...')

            self.ntokens = len(self.sentence_field.vocab)

            if pretrained_args is not None:

                #MUST USE EMBEDDINGS FROM PRETRAINED MODEL
                self.sentence_field.vocab.vectors = None

                if pretrained_args['hidden_size'] != self.hidden_size:
                    print('WARNING: pretrained model has a different hidden size, changing hidden size')
                    self.hidden_size = pretrained_args['hidden_size']

                if pretrained_args['wordvec_dim'] != self.wordvec_dim:
                    print('WARNING: pretrained model has a different embed dim, so ignoring embedding from pretrained model')
                    self.wordvec_dim = pretrained_args['wordvec_dim']

                '''
                if pretrained_args['num_layers'] != self.num_layers:
                    print('WARNING: pretrained model has more layers than specified, so setting number of layers to that of pretrained')
                    self.num_layers = pretrained_weights['num_layers']
                '''

            args = {'vocab_size' : self.ntokens,
                'num_classes' : self.num_classes,
                'batch_size' : self.batch_size,
                'cuda' : self.cuda,
                'vectors' : self.sentence_field.vocab.vectors,
                'input_size' : self.wordvec_dim,
                'dropout' : self.dropout,
                'rnn_dropout' : self.rnn_dropout,
                'hidden_size' : self.hidden_size,
                'num_layers' : self.num_layers,
                'train_word_vecs' : self.tune_wordvecs
            }

            #MAKING MATRIX TO SAVE ATTENTION WEIGHTS
            self.train_weights = {key: torch.zeros(len(self.train_data), self.max_length) for key in ['text', 'attn']}
            self.train_weights['hidden'] = torch.zeros(len(self.train_data), self.hidden_size)
            self.train_weights ['preds'] = torch.zeros(len(self.train_data))
            self.train_weights['targets'] = torch.zeros(len(self.train_data))
            if self.test_data is not None:
                self.test_weights = {key: torch.zeros(len(self.train_data), self.max_length) for key in ['text', 'attn']}
                self.test_weights['hidden'] = torch.zeros(len(self.train_data), self.hidden_size)
                self.test_weights['targets'] = torch.zeros(len(self.test_data))
                self.test_weights['preds'] = torch.zeros(len(self.test_data))


            #MAKING MATRIX TO SAVE ATTENTION WEIGHTS
            self.best_train_weights = {key: torch.zeros(len(self.train_data), self.max_length) for key in ['text', 'attn']}
            self.best_train_weights['hidden'] = torch.zeros(self.hidden_size)
            self.best_train_weights['preds'] = torch.zeros(len(self.train_data))
            self.best_train_weights['targets'] = torch.zeros(len(self.train_data))
            if self.test_data is not None:
                self.best_test_weights = {key: torch.zeros(len(self.train_data), self.max_length) for key in ['text', 'attn']}
                self.best_test_weights['hidden'] = torch.zeros(self.hidden_size)
                self.best_test_weights['targets'] = torch.zeros(len(self.test_data))
                self.best_test_weights['preds'] = torch.zeros(len(self.test_data))

            if self.attention_dim is None:
                self.model = VanillaRNN(**args)
                print('Using Vanilla RNN with following args:\n{}'
                        .format(args))

            else:
                print(self.attn_type)
                attn_args = {
                    'attention_dim' : self.attention_dim,
                    'tune_attn': self.tune_attn,
                    'attn_type' : self.attn_type,
                }
                args.update(attn_args)

                print('Using Attention model with following args:\n{}'
                        .format(args))
                self.model = SelfAttentiveRNN(**args)


            if self.cuda:
                self.model.cuda()

            if pretrained_weights is not None:
                self.model.init_pretrained(pretrained_weights, fix_pretrained = self.fix_pretrained)

        else:
            print('Loading Model from checkpoint')
            self.model = torch.load(self.checkpoint_path)


    def repackage_hidden(self, h):
        '''Wraps hidden states in new Variables, to detach them from their history.'''
        if type(h) == Variable:
            if self.cuda:
                #SAVE LABELS
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

                #GET ACCURACY
                preds = torch.max(predictions, dim = 1)[1]
                pct_correct = float(torch.sum(targets == preds)[0].data[0]/predictions.size(0))
                accuracies[i] = pct_correct

                if self.weight_saving:
                    #SAVING ATTENTION WEIGHTS
                    self.save_weights(i, data, A, h, preds, targets, "test")

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

                #GET ACCURACY
                preds = torch.max(predictions, dim = 1)[1]
                pct_correct = float(torch.sum(targets == preds)[0].data[0]/predictions.size(0))
                accuracies[i % self.log_interval] = pct_correct

                if self.weight_saving:
                    #SAVING ATTENTION WEIGHTS
                    self.save_weights(i, data, A, h, preds, targets, 'train')

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
                    elapsed = time() - start_time
                    accuracies = torch.zeros(self.log_interval)
                    print('At time: {elapsed} accuracy is {current_accuracy} and loss is {loss}'\
                            .format(elapsed=elapsed, current_accuracy = current_accuracy, loss = current_loss))

        return optimizer

    def start_train(self):
        print("Building RNN Classifier...")
        self.load_data()

        pretrained_vocab, pretrained_args, pretrained_weights = self.get_pretrained()

        self.get_vectors(pretrained_vocab)
        self.get_batches()
        self.get_model(pretrained_weights, pretrained_args)
        self.model.train()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = None
        if self.optim == 'adam':
            optimizer = Adam(parameters, lr = self.lr)
        elif self.optim == 'SGD':
            optimizer = SGD(parameters, lr = self.lr, weight_decay = self.l2)

        return optimizer


    def train(self, optimizer):

        start_time = time()
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
                print('Not better for {}'.format(not_better))

                if not_better >= 3:
                    if self.optim == 'vanilla_grad':
                        #Annealing
                        self.lr /= 4
                if not_better >= 10:
                    print('Model not improving. Stopping early'
                           'at {} epochs.'.format(self.best_accuracy, self.epoch))
                    break
            else:
                print('Achieved new best!')
                self.best_accuracy = self.eval_accuracy
                self.best_model = self.model
                self.best_train_weights = self.train_weights
                self.best_test_weights = self.test_weights
                not_better = 0

        print('Done Training. Achieved Best Accuracy of {}'.format(self.best_accuracy))

        if self.savepath is not None:

            self.save_checkpoint(self.savepath, optimizer)

            print('Finished Training.')

    def save_weights(self, i, text, attns, hidden, preds, targets, fold = 'train'):
        index = self.batch_size * i

        if fold == 'train':
            #SAVE TEXT
            self.train_weights['text'][index: index + self.batch_size, :text.size(1)] = text.data[:, :text.size(1)]
            #SAVE REPRESENTATIONS
            self.train_weights['hidden'][index: index + self.batch_size] = hidden.data
            if attns is not None:
                #SAVE ATTENTION WEIGHTS
                self.train_weights['attn'][index: index + self.batch_size, :attns.size(1)] = attns.data
            #SAVE PREDICTIONS
            self.train_weights['preds'][index: index + self.batch_size] = preds.data
            #SAVE CORRECT ANSWERS
            self.train_weights['targets'][index: index + self.batch_size] = targets.data
        elif fold == 'test':
            #SAVE TEXT
            self.test_weights['text'][index: index + self.batch_size, :text.size(1)] = text.data[:, :text.size(1)]
            if attns is not None:
                #SAVE ATTENTION WEIGHTS
                self.test_weights['attn'][index: index + self.batch_size, :attns.size(1)] = attns.data
            #SAVE REPRESENTATIONS
            self.test_weights['hidden'][index: index + self.batch_size] = hidden.data
            #SAVE PREDICTIONS
            self.test_weights['preds'][index: index + self.batch_size] = preds.data
            #SAVE CORRECT
            self.test_weights['targets'][index: index + self.batch_size] = targets.data


    def save_checkpoint(self, checkpointpath, optimizer = None, name = None):
        print("Saving Model Parameters and Results...")
        state = {
                    'epoch': self.epoch + 1,
                    'state_dict': self.best_model.state_dict(),
                    'best_valid_accuracy': self.best_accuracy,
                    'optimizer': None if optimizer is None else optimizer.state_dict(),
                    'accuracies': self.accuracies,
                    'vocab': self.sentence_field.vocab,
                    'labels': self.target_field.vocab,
                    'train_weights':self.best_train_weights,
                    'test_weights': self.best_test_weights
                }
        savepath = checkpointpath + ''.join(str(datetime.now()).split())
        if name is not None:
            savepath = checkpointpath + name
        print("Path:{}".format(savepath))

        torch.save(state, savepath)

    #def start_from_checkpoint(self, checkpoint):
        #current = torch.load(checkpoint)
        #self.model.load_


    def dump_attns(self, attn_path):
        if self.test_weights is not None:
            torch.save(self.test_weights, attn_path)

    def get_pretrained(self):

        if self.pretrained is not None:
            print('Using Pretrained RNN from path: {}'.format(self.pretrained))
            pretrained = torch.load(self.pretrained)
            pretrained_vocab = pretrained['vocab']
            pretrained_args = pretrained['args']
            pretrained_weights = pretrained['best_model']
            return pretrained_vocab, pretrained_args, pretrained_weights
        else:
            return None, None, None



if __name__ == '__main__':
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
    parser.add_argument('--tune_attn', type=str, default = TUNE_ATTN,
                        help='location of pretrained init')
    parser.add_argument('--glove_dim', type=int, default = GLOVE_DIM,
                        help='location of pretrained init')
    parser.add_argument('--wordvec_dim', type=int, default = WORDVEC_DIM,
                        help='location of pretrained init')
    parser.add_argument('--wordvec_source', type=str, default = WORDVEC_SOURCE,
                        help='location of pretrained init')
    parser.add_argument('--tune_wordvecs', type=str, default = TUNE_WORDVECS,
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
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    args.tune_wordvecs = str2bool(args.tune_wordvecs)
    args.tune_attn = str2bool(args.tune_attn)

    trainer = TrainClassifier(
                        num_classes = 2,
                        pretrained = args.pretrained,
                        checkpoint = args.checkpoint,
                        data = args.data,
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
                        tune_attn = args.tune_attn,
                        glove_dim = args.glove_dim,
                        wordvec_source = args.wordvec_source,
                        wordvec_dim = args.wordvec_dim,
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
                    )
    optimizer = trainer.start_train()
    trainer.train(optimizer)

