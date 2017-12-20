
params = {}

params['data'] = ['ptb', 'wikitext', 'gigasmall']
params['model'] = ['langmodel']
params['batch_size'] = [20, 50, 80]
params['seq_len'] = [20, 35, 50]
params['tune_wordvecs'] = [True, False]
params['vectors'] = ['', 'glove']
params['num_layers'] = [1,2,3]


combinations = []


for data in params['data']:
    for model in params['model']:
        for bsz in params['batch_size']:
            for seq in params['seq_len']:
                for tune in params['tune_wordvecs']:
                    for v in params['vectors']:
                        combinations.append({"data":data,
                            "model": model,
                            "batch_size": bsz,
                            "seq_len": seq,
                            "tune_wordvecs": tune,
                            "vectors": v
                        })
commands = ''

for each in combinations:
    args = ['--{} {}'.format(arg, each[arg]) for arg in each.keys()]
    all_args = ' '.join(args)
    command = 'python tune_models.py ' + all_args
    commands += command + '\n'

f = open('tunelangmodel.txt', 'w')
out = f.writelines(commands)


