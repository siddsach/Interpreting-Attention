import GPyOpt
import argparse


class Optimizer:
    def __init__(self, dataset, vectors, tune_wordvecs, wordvec_dim, choices, TrainerClass, timelimit, num_layers):

        print('Building Bayesian Optimizer for \n data:{} \n choices:{}'.format(dataset, choices))

        self.choices = choices
        self.dataset = dataset
        self.best_loss = 100000
        self.TrainerClass = TrainerClass
        self.model = None
        self.vectors = vectors
        self.runs = []
        self.num_layers = num_layers
        self.tune_wordvecs = tune_wordvecs
        self.wordvec_dim = wordvec_dim
        self.best_accuracy = 0.0

        myBopt = GPyOpt.methods.BayesianOptimization(f=self.getError,#Objective function
                                                            domain=choices,          # Box-constrains of the problem
                                                            initial_design_numdata = 5,   # Number data initial design
                                                            acquisition_type='EI',        # Expected Improvement
                                                            exact_feval = True
                                                        )



        myBopt.run_optimization(max_time = timelimit)

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

        settings["datapath"] = self.dataset
        settings["wordvec_source"] = self.vectors
        settings["num_layers"] = self.num_layers
        settings["tune_wordvecs"] = self.tune_wordvecs
        settings["wordvec_dim"] = self.wordvec_dim

        print("SETTINGS FOR THIS RUN")
        print(settings)
        print(self.TrainerClass)
        trainer = self.TrainerClass(**settings)
        trainer.train()
        if trainer.best_accuracy > self.best_accuracy:
            print('Improved accuracyfrom {} to {}'.format(self.accuracy, trainer.best_accuracy))
            self.best_loss = trainer.best_loss
            self.best_args = settings
            self.model = trainer
        self.runs.append({"params":settings, "loss": trainer.best_loss})

        return trainer.best_loss



    def test(params):
        print('testing')
        settings = {arg["name"]: value for arg, value in zip(params, params[0])}
        out = settings['lr']**2 + 5 * settings['batch_size']
        return out



# TRAIN LANGUAGE MODELS
LANGMODEL_CHOICES = [
    {"name":"lr", "type": "continuous", "domain":[0.00005, 0.005]},
    {"name":"batch_size", "type": "discrete", "domain": [20, 50, 80]},
    {"name":"seq_len", "type":"discrete", "domain":[20, 35, 50]},
    {"name":"dropout", "type": "continuous", "domain": [0,1]},
    {"name":"anneal", "type": "continuous", "domain": [2, 8]},
    {"name":"num_layers", "type": "discrete", "domain": [2, 3]}
]


CLASSIFIER_CHOICES = [
                {"name":"lr", "type": "continuous", "domain":[0.00005, 0.005]},
                {"name":"rnn_dropout", "type": "continuous", "domain": [0,1]},
                {"name":"dropout", "type": "continuous", "domain": [0,1]},
                {"name":"l2", "type": "continuous", "domain": [0, 1]},
            ]

ATTNCLF_CHOICES = [
                {"name": "attention_dim", "type": "discrete", "domain": [5, 100]},
                {"name": "mlp_hidden", "type": "discrete", "domain": [20, 500]}
            ]

MLPATTN_CHOICES = []

KEYVALATTN_CHOICES = []

def tuneModels(dataset, model, vectors, wordvec_dim, tune_wordvecs, num_layers):
    choices = None
    trainerclass = None

    if model == 'langmodel':
        choices = LANGMODEL_CHOICES
        from pretraining.langmodel.trainer import TrainLangModel
        trainerclass = TrainLangModel
    elif model == 'classifier':
        choices = CLASSIFIER_CHOICES
        from classifier.attention_rnn.trainer import TrainClassifier
        trainerclass = TrainClassifier
    elif model == 'mlpattn':
        choices = MLPATTN_CHOICES
        from classifier.attention_rnn.trainer import TrainClassifier
        trainerclass = TrainClassifier
    elif model == 'keyvalattn':
        choices = KEYVALATTN_CHOICES
        from classifier.attention_rnn.trainer import TrainClassifier
        trainerclass = TrainClassifier


    max_time = 3.5 * 60 * 60 ## maximum allowed time
    opt = Optimizer(dataset, vectors, tune_wordvecs, wordvec_dim, choices, trainerclass, max_time, num_layers)
    name = '{}_{}.pt'.format(dataset, vectors)
    folder = None
    if opt.finished:
        folder = 'optimized/'
    else:
        folder = 'optimizing/'

    #opt.model.save_checkpoint(folder + name)



parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
parser.add_argument('--data', type=str, default='ptb',
                    help='dataset')
parser.add_argument('--model', type=str, default='classifier',
                    help='type of model to train')
parser.add_argument('--vectors', type=str, default='',
                    help='vectors to use')
parser.add_argument('--num_layers', type=int, default=1,
                    help='vectors to use')
parser.add_argument('--tune_wordvecs', type=bool, default=True,
                    help='whether to tune wordvecs')
parser.add_argument('--wordvec_dim', type=int, default=200,
                    help='wordvec_dim')

args = parser.parse_args()

tuneModels(args.data, model = args.model, vectors = args.vectors, wordvec_dim = args.wordvec_dim,
             tune_wordvecs = args.tune_wordvecs, num_layers = args.num_layers)
