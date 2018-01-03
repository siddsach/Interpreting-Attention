import os
import json
import re
import numpy as np

all_results = {}
best_results = {}
for f in os.listdir():
    if f != 'mine_results.py':
        text = open(f, 'r').read()
        where = re.search('RESULTS:', text)
        if where is not None:
            begin = where.span()[1]
            nums = text[begin:]
            end = re.search(']', nums).span()[0]
            nums = nums[:end + 1]
            nums = nums.replace('\n', '')
            nums = nums.replace("'", "\"")
            nums = nums.replace('True', 'true')
            nums = nums.replace('False', 'false')
            data = json.loads(nums)
            for run in data:
                for param in run['params'].keys():
                    if param not in all_results.keys():
                        all_results[param] = []
                        best_results[param] = []
                    num = run['params'][param]
                    try:
                        num = float(num)
                    except:
                        pass



                    all_results[param].append(num)

                    if run['best_accuracy'] > -80:
                        best_results[param].append(num)

for key in all_results.keys():
    try:
        print('\n\n\n')
        print(key)
        print('\nall_results')
        print(np.mean(all_results[key]), np.std(all_results[key]))
        print('\nbest_results')
        print(np.mean(best_results[key]), np.std(best_results[key]))

    except:
        pass


