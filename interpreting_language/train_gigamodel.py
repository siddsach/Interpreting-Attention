from time import time
START = time()

from pretraining.langmodel.trainer import TrainLangModel
import subprocess
import os
import json
import math
import torch

current_path = os.getcwd()
data_dir = "data/gigaword"
timelimit = 3.75 * 60 * 60

max_size = 100000
path = "https://s3.amazonaws.com/gigaword/thread{}.txt"
total_files = 16
filenames= [path.format(i+1) for i in range(total_files)]
progresspath = 'trained_models/langmodel/gigaword/progress.json'
progress = json.load(open('trained_models/langmodel/gigaword/progress.json', 'w'))

def process(foldpath, progress):
    trainer = TrainLangModel()
    vocab = None
    params = None
    train_loss = None
    if progress["started"]:
        current = torch.load(current_path + '/trained_models/langmodel/gigaword/training/model.py')
        vocab = current["vocab"]
        params = current["state_dict"]
        train_loss = current["train_loss"]

    text = open(foldpath, 'r').read()
    trainer.current_loss = train_loss
    trainer.prepare_data(text, already_read = True, vocab = vocab)
    trainer.init_model(checkpoint_params = params)
    optimizer = trainer.train_step(None, trainer.model, START)

while time() - START < timelimit:
    current_thread = progress["current_thread"]
    if current_thread in progress["all_threads"].keys():
        current_fold = progress["all_threads"][current_thread]["current_fold"]
        foldpath = progress["all_threads"][current_thread]["folds"][current_fold]["path"]
        process(foldpath, progress)

        if current_fold == len(progress["all_threads"][current_thread]["folds"]) - 1:
            for fold in progress["all_threads"][current_thread]["folds"]:
                bashCommand = "rm " + fold["path"]
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                progress["current_thread"] += 1
        else:
            progress["all_threads"][current_thread]["current_fold"] += 1
    else:
        progress["all_threads"][current_thread] = {"folds":[], "current_fold": 0}
        path = filenames[current_thread]
        print('Downloading from {}...'.format(path))
        bashCommand = "wget " + path + " -P " + data_dir
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print('Reading and splitting file')
        size = os.path.getsize(data_dir + "/thread{}.txt".format(current_thread))
        num_folds = math.ceil(float(size) / max_size)
        reader = open(data_dir + "/thread{}.txt".format(current_thread + 1), 'r')
        docs = [el.split("\t") for el in reader.read().split("\n")]
        docs = [el[2] if len(el)==3 else "" for el in docs]
        split_size = len(docs) // num_folds

        progress["all_threads"][current_thread]["folds"] = []
        for i in range(0, num_folds):
            this_fold = docs[split_size * i: split_size * (i+1)]
            concatenated = "".join(text for text in this_fold)
            foldpath = data_dir + "/thread{}fold{}.txt".format(current_thread, i)
            foldfile = open(foldpath, 'w')
            foldfile.write(concatenated)

            progress["all_threads"][current_thread]["folds"].append({"finished":False, "path":foldpath})

        bashCommand = "rm " + data_dir + "/thread{}.txt".format(current_thread)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()



