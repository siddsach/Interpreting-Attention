from time import time
START = time()

from pretraining.langmodel.trainer import TrainLangModel
import subprocess
import os
import json
import math
import torch
import argparse

parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
parser.add_argument('--data',  type=str, default = 'reviews',
                    help='location of pretrained init')
ARGS = parser.parse_args()

current_path = os.getcwd()
data_dir = "data/reviews"
timelimit = 3.75 * 60 * 60

max_size = 1000000000
total_files = None
filenames = None
if ARGS.data == 'gigaword':
    path = "https://s3.amazonaws.com/gigaword/thread{}.txt"
    total_files = 16
    filenames= [path.format(i+1) for i in range(total_files)]
elif ARGS.data == 'reviews':
    total_files = 1
    filenames = ["https://s3.amazonaws.com/amazonmoviereviews/reviews.txt"]
else:
    assert False, "wrong data input"
progresspath = 'trained_models/langmodel/{}/progress.json'.format(ARGS.data)
try:
    print("RESUMING PROGRESS")
    progress = json.load(open(progresspath, 'r'))
except:
    print("STARTING NEW CHAIN")
    progress = {"started": False, "current_thread":0, "all_threads":{}}

savepath = current_path + '/trained_models/langmodel/{}/training/'.format(ARGS.data)

def run(foldpath, progress):
    trainer = TrainLangModel(savepath = savepath, time_limit = timelimit - (time() - START))
    vocab = None
    params = None
    train_loss = None
    if progress["started"]:
        current = torch.load(savepath + 'model.pt')
        vocab = current["vocab"]
        params = current["state_dict"]
        train_loss = current["train_loss"]
        args = current["args"]

    else:
        progress["started"] = True


    text = open(foldpath, 'r').read()
    print("TEXT LENGTH")
    print(len(text))
    trainer.current_loss = train_loss
    trainer.prepare_data(text, already_read = True, vocab = vocab)
    trainer.init_model(checkpoint_params = params)
    print(trainer.model)
    optimizer = trainer.train_step(None, trainer.model, START)
    trainer.save_checkpoint()
    return progress

while time() - START < timelimit:
    current_thread = progress["current_thread"]
    if current_thread in progress["all_threads"].keys():
        current_fold = progress["all_threads"][current_thread]["current_fold"]
        print('CURRENT:{}'.format(time() - START))
        if time() - START < timelimit:
            foldpath = progress["all_threads"][current_thread]["folds"][current_fold]["path"]
            print("FOLDPATH:{}".format(foldpath))
            progress = run(foldpath, progress)

            if current_fold == len(progress["all_threads"][current_thread]["folds"]) - 1:
                for fold in progress["all_threads"][current_thread]["folds"]:
                    bashCommand = "rm " + fold["path"]
                    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                    output, error = process.communicate()
                    progress["current_thread"] += 1
            else:
                progress["all_threads"][current_thread]["current_fold"] += 1
        else:
            print("REACHED TIMILIMIT, QUITTING TRAINING")
    else:
        progress["all_threads"][current_thread] = {"folds":[], "current_fold": 0}
        path = filenames[current_thread]
        print('Downloading from {}...'.format(path))
        bashCommand = "wget " + path + " -P " + data_dir
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print('Reading and splitting file')
        reader = None
        bashCommand = None
        if ARGS.data == 'gigaword':
            filename = data_dir + "/thread{}.txt".format(current_thread + 1)
            size = os.path.getsize(filename)
            num_folds = math.ceil(float(size) / max_size)
            reader = open(filename, 'r')
            docs = [el.split("\t") for el in reader.read().split("\n")]
            docs = [el[2] if len(el)==3 else "" for el in docs]
            split_size = len(docs) // num_folds
            progress["all_threads"][current_thread]["folds"] = []
            for i in range(0, num_folds):
                this_fold = docs[split_size * i: split_size * (i+1)]
                concatenated = "".join(text for text in this_fold)
                foldpath = data_dir + "/fold{}.txt".format(i)
                foldfile = open(foldpath, 'w')
                foldfile.write(concatenated)
                progress["all_threads"][current_thread]["folds"].append({"path":foldpath})

            bashCommand = "rm " + data_dir + "/thread{}.txt".format(current_thread + 1)
        elif ARGS.data == 'reviews':
            filename = data_dir + "/reviews.txt"
            size = os.path.getsize(filename)
            num_folds = math.ceil(float(size) / max_size)
            text = open(filename, 'r').read()
            fold_size = len(text) // num_folds
            for i in range(0, num_folds):
                this_fold = text[i*fold_size:(i+1)*fold_size]
                foldpath = data_dir + "/fold{}.txt".format(i)
                foldfile = open(foldpath, 'w')
                foldfile.write(this_fold)

                progress["all_threads"][current_thread]["folds"].append({"path":foldpath})

            bashCommand = "rm " + data_dir + "/reviews.txt".format(current_thread + 1)



        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    json.dump(progress, open(progresspath, "w"))



