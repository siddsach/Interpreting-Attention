import argparse
import torch
from pretraining.langmodel.trainer import TrainLangModel
import os

current_path = os.getcwd()
project_path = current_path#[:len(current_path)-len('/pretraining/langmodel')]

TIME_LIMIT = 3 * 60 * 60
DATASET = 'ptb'
WIKI_PATH = project_path + '/data/wikitext-2/wikitext-2/'
PTB_PATH = project_path + '/data/penn/'
GIGA_PATH = project_path + '/data/gigaword/'
MODEL_SAVE_PATH = project_path + '/trained_models/langmodel/'
VECTOR_CACHE = project_path + '/vectors'

NUM_EPOCHS = 1 if not torch.cuda.is_available() else 5
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
TIE_WEIGHTS = False
MODEL_TYPE = 'LSTM'
OPTIMIZER = 'vanilla_grad'
DROPOUT = 0.4
RNN_DROPOUT = 0.2
HIDDEN_SIZE = 1024
FEW_BATCHES = 1000 if not torch.cuda.is_available() else None
MAX_VOCAB = 100000
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
