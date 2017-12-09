import GPy
import GPyOpt
from pretraining.langmodel.trainer import TrainLangModel
from classifier.attention_rnn.trainer import TrainClassifier
import json
import argparse


def optimize(dataset, choices, TrainerClass):

    def getError(params):
        settings = {}
        for arg, value in zip(choices, params[0]):
            value = value.item()
            if arg["type"] == "discrete":
                value = int(value)

            settings[arg['name']] = value

        settings["data"] = dataset

        print("SETTINGS FOR THIS RUN")
        print(settings)
        trainer = TrainerClass(**settings)
        trainer.train()
        return trainer.best_objective

    def test(params):
        print('testing')
        settings = {arg["name"]: value for arg, value in zip(params, params[0])}
        out = settings['lr']**2 + 5 * settings['batch_size']
        return out



    myBopt = GPyOpt.methods.BayesianOptimization(f=getError,#Objective function
                                                        domain=choices,          # Box-constrains of the problem
                                                        initial_design_numdata = 5,   # Number data initial design
                                                        acquisition_type='EI',        # Expected Improvement
                                                        exact_feval = True
                                                    )



    max_iter = 1 ## maximum number of iterations
    max_time = 3.5 * 60 * 60 ## maximum allowed time

    myBopt.run_optimization(max_time = max_time, max_iter = max_iter)
    return myBopt.x_opt, myBopt.fx_opt


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
    best = {}

    if model == 'langmodel':
        best_params, best_loss = optimize(dataset, LANGMODEL_CHOICES, TrainLangModel)
    elif model == 'classifier':
        best_params, best_loss = optimize(dataset, CLASSIFIER_CHOICES, TrainClassifier)
    elif model == 'mlpattn':
        best_params, best_loss = optimize(dataset, MLPATTN_CHOICES, TrainClassifier)
    elif model == 'keyvalattn':
        best_params, best_loss = optimize(dataset, KEYVALATTN_CHOICES, TrainClassifier)

    best['params'] = best_params
    best['loss'] = best_loss

    if savepath is None:
        savepath = '{}_{}.json'.format(model, dataset)

    json.dump(best, open(savepath, 'w'))


parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
parser.add_argument('--data', type=str, default='ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='langmodel',
                    help='location of the data corpus')
parser.add_argument('--savepath', type=str, default=None,
                    help='location of the data corpus')

args = parser.parse_args()

tuneModels(args.data, model = args.model, savapeth = args.savepath)
