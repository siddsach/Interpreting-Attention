import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt

discrete_params = ["tune_wordvecs", "num_layers", 'tune_attn']#, "wordvec_source", "attn_type"]
continuous_params = ["lr", "dropout", "rnn_dropout", "l2"]
results = {"discrete": {param:{} for param in discrete_params}, "continuous":{param:{"choice":[], "result":[]} for param in continuous_params}}
dataset = "MPQA"
vectors = "google"
attn_type = 'similarity'

def consider(params, dataset, vectors, attn_type):
    if dataset is not None:
        if params["data"] != dataset:
            return False
    if vectors is not None:
        if params["wordvec_source"] != vectors:
            return False
    if attn_type is not None:
        if params["attn_type"] != attn_type:
            print(params["attn_type"], attn_type)
            return False
    return True


def f(best_results):
    for folder in os.listdir():

        if folder != 'parse_results.py':
            for out in os.listdir(folder):
                path = folder + "/" + out
                text = open(path, 'r').read()
                match = re.search(r'(RESULTS:\n)(.*)(\n)',text)
                if match is not None:
                    data = match.group(2)
                    data = data.replace("'", "\"")
                    data = data.replace("True", "true")
                    data = data.replace("False", "false")
                    data = data.replace("None", "0")
                    data = data.replace("datapath", "data")
                    data = json.loads(data)
                    for run in data:
                        if 'attn_type' in run['params'].keys():
                            if consider(run["params"], dataset, vectors, attn_type):
                                best_results.append(run)
                                print('here')
                                for param in run["params"].keys():
                                    if param in discrete_params:
                                        print("DISCRETE")
                                        print(param)
                                        if run["params"][param] not in results["discrete"][param].keys():
                                            results["discrete"][param][run["params"][param]] = []
                                        results["discrete"][param][run["params"][param]].append(run["best_accuracy"])
                                    elif param in continuous_params:
                                        print("CONTINOUS")
                                        print(param)
                                        results["continuous"][param]["choice"].append(run["params"][param])
                                        results["continuous"][param]["result"].append(run["best_accuracy"])
                                    elif param != "data":
                                        print("NOT FOUND")
                                        print(param)

    return best_results

if __name__ == '__main__':
    best_results = []
    best_results = f(best_results)

    best_results = sorted(best_results, reverse = True, key = lambda k: k["best_accuracy"])[:10]
    for run in best_results:
        print("ACCURACY")
        print(run["best_accuracy"])
        print("PARAMS")
        print(run['params'])


    for i, param in enumerate(results['discrete'].keys()):
        fig = plt.figure()
        ax = fig.add_subplot((i+1)*100 + 11)
        data = []
        names = []
        print(param)
        for choice in results['discrete'][param].keys():
            data.append(results['discrete'][param][choice])
            names.append(choice)
        ax.boxplot(data)
        ax.set_title(param)
        ax.set_xticklabels(names)
        plt.show()

    for i, param in enumerate(results['continuous'].keys()):
        fig = plt.figure()
        ax = fig.add_subplot((i+1)*100 + 11)
        ax.set_xlabel(param)
        ax.set_ylabel("accuracy")
        ax.scatter(results['continuous'][param]['choice'], results['continuous'][param]['result'])
        ax.set_ylim(0.7, 0.85)
        plt.show()









