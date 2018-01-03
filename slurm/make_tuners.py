
params = {}

params['data'] = ['IMDB', 'MPQA']
params['model'] = ['classifier', 'mlpattn', 'keyvalattn']
params['batch_size'] = [32]
params['tune_wordvecs'] = ["True", "False"]
params['vectors'] = ['None', 'glove', 'google', 'gigavec']
params['num_layers'] = [1,2,3]
params['tie_weights'] = ["True", "False"]
params['tune_attn'] = ["True", "False"]



combinations = []


for data in params['data']:
    for model in params['model']:
        for bsz in params['batch_size']:
            for choice in params['tune_attn']:
                for tune in params['tune_wordvecs']:
                    for v in params['vectors']:
                        for n in params['num_layers']:
                            combinations.append({"data":data,
                                "model": model,
                                "batch_size": bsz,
                                "tune_attn": choice,
                                "tune_wordvecs": tune,
                                "vectors": v,
                                "num_layers": n
                            })
commands = ''

for each in combinations:
    args = ['--{} {}'.format(arg, each[arg]) for arg in each.keys()]
    all_args = ' '.join(args)
    command = 'python tune_models.py ' + all_args
    commands += command + '\n'

f = open('tuneclf.txt', 'w')
out = f.writelines(commands)


