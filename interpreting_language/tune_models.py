import GPy
import GPyOpt
from pretraining.langmodel.trainer import TrainLangModel

vanilla_params = [
    {"name":"lr", "type": "continuous", "domain": [0, 10]},
    {"name":"batch_size", "type": "discrete", "domain": [10, 100]},
    {"name":"seq_len", "type":"discrete", "domain":[10, 100]},
    {"name":"dropout", "type": "continuous", "domain": [0,1]},
    {"name":"anneal", "type": "continuous", "domain": [2, 8]},
    {"name":"num_layers", "type": "discrete", "domain": [1, 2, 3]}
]

def getError(params):
    settings = {}
    for arg, value in zip(vanilla_params, params[0]):
        value = value.item()
        if arg["type"] == "discrete":
            value = int(value)

        settings[arg['name']] = value

    print("SETTINGS FOR THIS RUN")
    print(settings)
    trainer = TrainLangModel(**settings)
    trainer.train()
    return trainer.best_eval_perplexity

def test(params):
    print('testing')
    settings = {arg["name"]: value for arg, value in zip(vanilla_params, params[0])}
    out = settings['lr']**2 + 5 * settings['batch_size']
    return out


attn_params = vanilla_params + \
                [{"name": "attention_dim", "type": "discrete", "domain": [5, 100]},
                 {"name": "mlp_hidden", "type": "discrete", "domain": [20, 500]}]


myBopt = GPyOpt.methods.BayesianOptimization(f=getError,#Objective function
                                                    domain=vanilla_params,          # Box-constrains of the problem
                                                    initial_design_numdata = 5,   # Number data initial design
                                                    acquisition_type='EI',        # Expected Improvement
                                                    exact_feval = True
                                                )



max_iter = 20       ## maximum number of iterations
max_time = 3.5 * 60 * 60 ## maximum allowed time
eps      = 0.001 ## tolerance, max distance between consicutive evaluations.

myBopt.run_optimization(max_time,eps)
