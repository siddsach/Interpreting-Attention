import GPyOpt
import argparse
from time import time

# TRAIN LANGUAGE MODELS
LANGMODEL_CHOICES = [
    {"name":"lr", "type": "continuous", "domain":[0.00005, 0.005]},
    {"name":"dropout", "type": "continuous", "domain": [0,1]},
    {"name":"anneal", "type": "continuous", "domain": [2, 8]},
]


CLASSIFIER_CHOICES = [
                {"name":"lr", "type": "continuous", "domain":[0.00005, 0.005]},
                {"name":"rnn_dropout", "type": "continuous", "domain": [0,1]},
                {"name":"dropout", "type": "continuous", "domain": [0,1]},
                {"name":"l2", "type": "continuous", "domain": [0, 1]},
            ]

ATTNCLF_CHOICES = [
                {"name": "attention_dim", "type": "attention_dim", "domain": [5, 100]}
            ]

class Optimizer:
    def __init__(self, dataset, model, vectors, wordvec_dim, tune_wordvecs, tune_attn, num_layers, timelimit, \
            batch_size, seq_len):

        self.start_time = time()

        self.choices = None
        self.TrainerClass= None

        if model == 'langmodel':
            self.choices = LANGMODEL_CHOICES
            from pretraining.langmodel.trainer import TrainLangModel
            self.TrainerClass = TrainLangModel
        elif model == 'classifier' or model == 'keyvalattn':
            self.choices = CLASSIFIER_CHOICES
            from classifier.attention_rnn.trainer import TrainClassifier
            self.TrainerClass = TrainClassifier
        elif model == 'mlpattn':
            self.choices = CLASSIFIER_CHOICES + ATTNCLF_CHOICES
            from classifier.attention_rnn.trainer import TrainClassifier
            self.TrainerClass = TrainClassifier

        print('Building Bayesian Optimizer for \n data:{} \n choices:{}'.format(dataset, self.choices))

        self.dataset = dataset
        self.best_loss = 100000
        self.model = None
        self.vectors = vectors
        self.num_layers = num_layers
        self.tune_wordvecs = tune_wordvecs
        self.tune_attn = tune_attn
        self.wordvec_dim = wordvec_dim
        self.best_accuracy = -10000000
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timelimit = timelimit

        self.runs = []
        self.model = None

        myBopt = GPyOpt.methods.BayesianOptimization(f=self.getError,#Objective function
                                                            domain=self.choices,          # Box-constrains of the problem
                                                            initial_design_numdata = 5,   # Number data initial design
                                                            acquisition_type='EI',        # Expected Improvement
                                                            exact_feval = True
                                                        )

        myBopt.run_optimization(max_iter = 1, max_time = timelimit)

        self.save()


    def save(self):

        savepath = '{}/optimized/{}bsz_{}seq_len_{}layers_{}vectors_{}tune_{}accuracy.pt'.format(self.dataset, self.batch_size, self.seq_len, self.num_layers, self.vectors, self.tune_wordvecs, self.best_accuracy)

        self.model.save_checkpoint(savepath)

        print("\n\n\nRESULTS:\n{}".format(self.runs))

    def getError(self, params):
        settings = {}
        for arg, value in zip(self.choices, params[0]):
            value = value.item()
            if arg["type"] == "discrete":
                value = int(value)
            elif arg["type"] == "continuous":
                value = float(value)

            settings[arg['name']] = value

        settings["data"] = self.dataset
        settings["wordvec_source"] = self.vectors
        settings["num_layers"] = self.num_layers
        settings["tune_wordvecs"] = self.tune_wordvecs
        settings["wordvec_dim"] = self.wordvec_dim
        settings["batch_size"] = self.batch_size

        if self.model == 'langmodel':
            settings["seq_len"] = self.seq_len
        elif self.model == ['keyvalattn']:
            settings["tune_attn"] = self.tune_attn

        print("SETTINGS FOR THIS RUN")
        print(settings)
        trainer = self.TrainerClass(**settings)
        optimizer = trainer.start_train()
        trainer.train(optimizer)

        if trainer.best_accuracy > self.best_accuracy:
            print('Improved accuracyfrom {} to {}'.format(self.best_accuracy, trainer.best_accuracy))
            self.best_accuracy = trainer.best_accuracy
            self.best_args = settings
            self.model = trainer

        print(self.model)
        self.runs.append({"params":settings, "best_accuracy": trainer.best_accuracy})

        current_time = time()
        if current_time - self.start_time >= self.timelimit:
            self.save()

        return -trainer.best_accuracy



    def test(params):
        print('testing')
        settings = {arg["name"]: value for arg, value in zip(params, params[0])}
        out = settings['lr']**2 + 5 * settings['batch_size']
        return out




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
    parser.add_argument('--data', type=str, default='ptb',
                        help='dataset')
    parser.add_argument('--model', type=str, default='classifier',
                        help='type of model to train')
    parser.add_argument('--vectors', type=str, default='',
                        help='vectors to use')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='vectors to use')
    parser.add_argument('--tune_wordvecs', type=str, default='True',
                        help='whether to tune wordvecs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='wordvec_dim')
    parser.add_argument('--wordvec_dim', type=int, default=200,
                        help='wordvec_dim')
    parser.add_argument('--timelimit', type=int, default=3*60*60,
                        help='wordvec_dim')
    parser.add_argument('--seq_len', type=int, default=35,
                        help='wordvec_dim')
    parser.add_argument('--tune_attn', type=str, default='true',
                        help='wordvec_dim')

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

    Optimizer(dataset = args.data, model = args.model, vectors = args.vectors, wordvec_dim = args.wordvec_dim,
                 tune_wordvecs = args.tune_wordvecs, tune_attn = args.tune_attn, num_layers = args.num_layers, timelimit = args.timelimit,
                 batch_size = args.batch_size, seq_len = args.seq_len)

