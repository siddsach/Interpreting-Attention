import GPyOpt
import argparse


class Optimizer:
    def __init__(self, dataset, vectors, choices, TrainerClass, timelimit):

        print('Building Bayesian Optimizer for \n data:{} \n choices:{}'.format(dataset, choices))

        self.choices = choices
        self.dataset = dataset
        self.best_loss = 100000
        self.TrainerClass = TrainerClass
        self.model = None
        self.vectors = vectors
        self.runs = []

        myBopt = GPyOpt.methods.BayesianOptimization(f=self.getError,#Objective function
                                                            domain=choices,          # Box-constrains of the problem
                                                            initial_design_numdata = 5,   # Number data initial design
                                                            acquisition_type='EI',        # Expected Improvement
                                                            exact_feval = True
                                                        )



        myBopt.run_optimization(max_time = timelimit)

        print("ATTRIBUTES")
        print(myBopt.__dict__)

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

        print("SETTINGS FOR THIS RUN")
        print(settings)
        trainer = self.TrainerClass(**settings)
        trainer.train()
        if trainer.best_loss < self.best_loss:
            print('Improved loss from {} to {}'.format(self.best_loss, trainer.best_loss))
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
    {"name":"learning_rate", "type": "continuous", "domain":[2, 30]},
    {"name":"batch_size", "type": "discrete", "domain": [20, 50, 80]},
    {"name":"seq_len", "type":"discrete", "domain":[20, 35, 50]},
    {"name":"dropout", "type": "continuous", "domain": [0,1]},
    {"name":"anneal", "type": "continuous", "domain": [2, 8]},
    {"name":"num_layers", "type": "discrete", "domain": [2, 3]}
]


CLASSIFIER_CHOICES = [
                {"name": "attention_dim", "type": "discrete", "domain": [5, 100]},
                {"name": "mlp_hidden", "type": "discrete", "domain": [20, 500]},
            ]

MLPATTN_CHOICES = []

KEYVALATTN_CHOICES = []

def tuneModels(dataset, model, vectors):
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

    vecs = []
    if vectors == 'glove':
        vecs = ['GloVe']
    elif vectors == 'charlevel':
        vecs = ['GloVe', 'charLevel']
    elif vectors == 'google':
        vecs = ['googlenews']

    max_time = 3.5 * 60 * 60 ## maximum allowed time
    opt = Optimizer(dataset, vecs, choices, trainerclass, max_time)
    name = '{}_{}.pt'.format(dataset, vectors)
    folder = None
    if opt.finished:
        folder = 'optimized/'
    else:
        folder = 'optimizing/'

    opt.model.save_checkpoint(folder + name)



parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
parser.add_argument('--data', type=str, default='ptb',
                    help='dataset')
parser.add_argument('--model', type=str, default='langmodel',
                    help='type of model to train')
parser.add_argument('--vectors', type=str, default='glove',
                    help='vectors to use')

args = parser.parse_args()

tuneModels(args.data, model = args.model, vectors = args.vectors)
