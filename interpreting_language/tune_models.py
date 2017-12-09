import GPy
import GPyOpt
from pretraining.langmodel.trainer import TrainLangModel
from classifier.attention_rnn.trainer import TrainClassifier
import json
import argparse


class Optimizer:
    def __init__(self, dataset, choices, TrainerClass):

        print('Building Bayesian Optimizer for \n data:{} \n choices:{}'.format(dataset, choices))

        self.choices = choices
        self.dataset = dataset
        self.best_loss = 100000
        self.TrainerClass = TrainerClass
        self.model = None

        myBopt = GPyOpt.methods.BayesianOptimization(f=self.getError,#Objective function
                                                            domain=choices,          # Box-constrains of the problem
                                                            initial_design_numdata = 5,   # Number data initial design
                                                            acquisition_type='EI',        # Expected Improvement
                                                            exact_feval = True
                                                        )



        max_iter = 1 ## maximum number of iterations
        max_time = 3.5 * 60 * 60 ## maximum allowed time

        myBopt.run_optimization(max_time = max_time, max_iter = max_iter)

        print("ATTRIBUTES")
        print(myBopt.__dict__)
        self.model.save_checkpoint()

    def getError(self, params):
        settings = {}
        for arg, value in zip(self.choices, params[0]):
            value = value.item()
            if arg["type"] == "discrete":
                value = int(value)

            settings[arg['name']] = value

        settings["data"] = self.dataset

        print("SETTINGS FOR THIS RUN")
        print(settings)
        trainer = self.TrainerClass(**settings)
        trainer.train()
        if trainer.best_loss < self.best_loss:
            self.best_loss = trainer.best_loss
            self.model = trainer
        return trainer.best_loss



    def test(params):
        print('testing')
        settings = {arg["name"]: value for arg, value in zip(params, params[0])}
        out = settings['lr']**2 + 5 * settings['batch_size']
        return out



# TRAIN LANGUAGE MODELS
LANGMODEL_CHOICES = [
    {"name":"batch_size", "type": "discrete", "domain": [20, 50, 80, 100]},
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

def tuneModels(dataset, model, savepath):
    choices = None
    trainerclass = None

    if model == 'langmodel':
        choices = LANGMODEL_CHOICES
        trainerclass = TrainLangModel
    elif model == 'classifier':
        choices = CLASSIFIER_CHOICES
        trainerclass = TrainClassifier
    elif model == 'mlpattn':
        choices = MLPATTN_CHOICES
        trainerclass = TrainClassifier
    elif model == 'keyvalattn':
        choices = KEYVALATTN_CHOICES
        trainerclass = TrainClassifier

    opt = Optimizer(dataset, choices, trainerclass)




parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
parser.add_argument('--data', type=str, default='ptb',
                    help='dataset')
parser.add_argument('--model', type=str, default='langmodel',
                    help='type of model to train')
parser.add_argument('--savepath', type=str, default=None,
                    help='where to save everything')

args = parser.parse_args()

tuneModels(args.data, model = args.model, savepath = args.savepath)
