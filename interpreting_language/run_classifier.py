import argparse
import torch
from classifier.attention_rnn.trainer import TrainClassifier
import os
#from visualize import plot_attn

root_path = os.getcwd()
print("ROOT_PATH: {}".format(root_path))

#### DEFAULTS ####
SPLIT = 0.75
DATASET = 'IMDB'
# 'sentence_subjectivity.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
IMDB_PATH = root_path + '/data/imdb/aclImdb'
MPQA_PATH = root_path + '/data/mpqa/mpqa_subj_labels.pickle'
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

MAX_DATA_LEN = 1000
if torch.cuda.is_available():
    MAX_DATA_LEN = None
def sorter(example):
    return len(example.text)

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
parser.add_argument('--fix_pretrained', type=int, default=None,
                    help='how many layers to fix')
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
                    l2 = args.l2,
                    fix_pretrained = args.fix_pretrained
                )
optimizer = trainer.start_train()
trainer.train(optimizer)
trainer.save_checkpoint('', optimizer, name = 'clf4attn.pt')

trained = torch.load('clf4attn.pt')
vocab = trained['vocab']
train_attns = trained['train_attns']

numrows = train_attns[0].size(0)
datawords = numrows * ['']
datavals = numrows * ['']

for i in range(numrows):
    words = [vocab.itos[i] for i in train_attns[0][i] if i!=0]
    datawords[i] = words
    vals = [i for i in train_attns[1][i] if i < len(words)]
    datavals[i] = vals

#plot_attn(datawords, datavals, savepath = 'test.png')


if args.attention is not None:
    print(trainer.train_attns)

