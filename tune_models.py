import GPy
import GPyOpt
from classifier.attention_rnn import TrainClassifier

def getClassifierError(params):
    trainer = Trainer.TrainClassifier(params)
    trainer.train()
    return trainer.eval_loss


vanilla_params = [
    {"name":"lr", "type": "continuous", domain: [0, 10]},
    {"name":"dropout", "type": "continuous", domain: [0, 0.7]}
]

attn_params = vanilla_params + \
                [{"name": "attention_dim", "type": "discrete", domain: [5, 100]},
                 {"name": "mlp_hidden", "type": "discrete", domain: [20, 500]}]


BBX_optimizer == GPyOpt.methods.BayesianOptimization(f=getClassifierError(,# Objective function
                                                    domain=mixed_domain,          # Box-constrains of the problem
                                                    initial_design_numdata = 5,   # Number data initial design
                                                    acquisition_type='EI',        # Expected Improvement
                                                    exact_feval = True
                                                )



