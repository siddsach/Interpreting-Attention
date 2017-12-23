
params = {}

params['data'] = ['ptb', 'wikitext']
params['model'] = ['langmodel']
params['batch_size'] = [32]
params['seq_len'] = [35]
params['tune_wordvecs'] = [True, False]
params['vectors'] = ['None', 'glove', 'google', 'gigavec']
params['num_layers'] = [1,2,3]
params['tie_weights'] = [True, False]


combinations = []


for data in params['data']:
    for model in params['model']:
        for bsz in params['batch_size']:
            for seq in params['seq_len']:
                for tune in params['tune_wordvecs']:
                    for tie in params['tie_weights']:
                        for v in params['vectors']:
                            for n in params['num_layers']:
                                combinations.append({"data":data,
                                    "model": model,
                                    "batch_size": bsz,
                                    "seq_len": seq,
                                    "tune_wordvecs": tune,
                                    "vectors": v,
                                    "tie_weights": tie,
                                    "num_layers": n
                                })
commands = ''

for each in combinations:
    args = ['--{} {}'.format(arg, each[arg]) for arg in each.keys()]
    all_args = ' '.join(args)
    command = 'python tune_models.py ' + all_args
    commands += command + '\n'

f = open('tunelangmodel.txt', 'w')
out = f.writelines(commands)


