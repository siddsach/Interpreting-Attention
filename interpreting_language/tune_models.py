import GPy
import GPyOpt
from pretraining.langmodel.trainer import TrainLangModel

def getClassifierError(params):
    print('INSIDE')
    print(params)

    trainer = TrainLangModel(params)
    trainer.train()
    return trainer.eval_loss


vanilla_params = [
    {"name":"lr", "type": "continuous", "domain": [0, 10]},
    {"name":"batch_size", "type": "discrete", "domain": [10, 100]}
]

attn_params = vanilla_params + \
                [{"name": "attention_dim", "type": "discrete", "domain": [5, 100]},
                 {"name": "mlp_hidden", "type": "discrete", "domain": [20, 500]}]


myBopt = GPyOpt.methods.BayesianOptimization(f=getClassifierError,#Objective function
                                                    domain=vanilla_params,          # Box-constrains of the problem
                                                    initial_design_numdata = 5,   # Number data initial design
                                                    acquisition_type='EI',        # Expected Improvement
                                                    exact_feval = True
                                                )



max_iter = 20       ## maximum number of iterations
max_time = 60       ## maximum allowed time
eps      = 1e-6     ## tolerance, max distance between consicutive evaluations.

myBopt.run_optimization(max_iter,eps=0)
