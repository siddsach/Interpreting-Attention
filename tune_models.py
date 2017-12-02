import GPy
import GPyOpt
from classifier.attention_rnn import TrainClassifier

def getClassifierError(params):
    trainer = Trainer.TrainClassifier(params)
    trainer.train()
    return trainer.eval_loss


params = [
    {"name":"lr", "type": "continuous", domain: [0, 10]},
    {"name":"dropout". "type": "continuous", domain: [0, 0.7]}
]


    myBopt = GPyOpt.methods.BayesianOptimization(f=getClassifierError(,# Objective function
    domain=mixed_domain,          # Box-constrains of the problem
    initial_design_numdata = 5,   # Number data initial design
    acquisition_type='EI',        # Expected Improvement
exact_feval = True)



