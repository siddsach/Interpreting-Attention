import re
import os
import json
import matplotlib.pyplot as plt

discrete_params = ["tune_wordvecs", "num_layers", "wordvec_source", "wordvec_dim"]
continuous_params = ["lr", "dropout", "rnn_dropout", "l2"]
results = {"discrete": {param:{} for param in discrete_params}, "continuous":{param:{"choice":[], "result":[]} for param in continuous_params}}
dataset = "IMDB"


for out in os.listdir():
    text = open(out, 'r').read()
    match = re.search(r'(RESULTS:\n)(.*)(\n)',text)
    if match is not None:
        data = match.group(2)
        #try:
        data = data.replace("'", "\"")
        data = data.replace("True", "true")
        data = data.replace("False", "false")
        data = json.loads(data)
        for run in data:
            if run["params"]["datapath"] == dataset:
                for param in run["params"].keys():
                    if param in discrete_params:
                        if run["params"][param] not in results["discrete"][param].keys():
                            results["discrete"][param][run["params"][param]] = []
                        results["discrete"][param][run["params"][param]].append(run["best_accuracy"])
                    elif param in continuous_params:
                        results["continuous"][param]["choice"].append(run["params"][param])
                        results["continuous"][param]["result"].append(run["best_accuracy"])
                    elif param != "datapath":
                        print("NOT FOUND")
                        print(param)

for param in results['discrete'].keys():
    print("PARAM")
    print(param)
    for choice in results['discrete'][param].keys():
        print(choice)
        data = results['discrete'][param][choice]
        print("MEAN:{} STD:{}".format(np.mean(data), np.std(data)))





        '''
        except:
            print("READ ERROR")
            print(data)
        '''

