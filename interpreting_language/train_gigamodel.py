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

path = "https://s3.amazonaws.com/gigaword/thread{}.txt"
total_files = 16
filenames= [path.format(i+1) for i in range(total_files)]

class Sequence:
    def __init__(self,
                timelimit = 3.75 * 60 * 60,
                max_size = 100000,
                dataset = 'gigaword',
                thread_paths = filenames
            ):
        current_path = os.getcwd()
        self.timelimit = timelimit
        self.max_size = max_size

        if dataset == 'gigaword':
            self.data_dir = 'data/gigaword'
            path = "https://s3.amazonaws.com/gigaword/thread{}.txt"
            total_files = 16
            self.thread_paths = [path.format(i+1) for i in range(total_files)]
        elif dataset == 'reviews':
            self.data_dir = 'data/reviews'
            total_files = 1
            self.thread_paths = ["https://s3.amazonaws.com/amazonmoviereviews/reviews.txt"]

        self.dataset = dataset
        trained_path = 'trained_models/langmodel'
        progresspath = trained_path + '/{}/progress.json'.format(dataset)
        if os.path.exists(progresspath):
            self.progress = json.load(open(progresspath, 'r'))
        else:
            self.progress = {"all_threads":{}, "current_thread":0, "iterations": 0}


        self.savepath = current_path + '/trained_models/langmodel/{}/training/'.format(self.data)

        self.run()
        json.dump(self.progress, open(progresspath, 'w'))

    def run_fold(self, current_thread):
        current_fold = self.progress["all_threads"][current_thread]["current_fold"]
        foldpath = self.progress["all_threads"][current_thread]["folds"][current_fold]["path"]
        self.process(foldpath, self.progress)
        finished_folds = current_fold == len(self.progress["all_threads"][current_thread]["folds"]) - 1
        if finished_folds:
            for fold in self.progress["all_threads"][current_thread]["folds"]:
                bashCommand = "rm " + fold["path"]
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
            self.progress["current_thread"] += 1
        else:
            self.progress["all_threads"][current_thread]["current_fold"] += 1

    def run(self):
        current_thread = self.progress["current_thread"]
        already_downloaded = current_thread in self.progress["all_threads"].keys()
        if not already_downloaded:
            docs = self.download_thread(current_thread)
            self.make_folds(docs, current_thread)

        self.run_fold(current_thread)

        if self.progress["current_thread"] >= len(self.thread_paths):
            print("FINISHED ITERATION, REINITING CYCLE")
            self.progress["iterations"] += 1
            self.progress["current_thread"] = 0
            for thread in self.progress["all_threads"].keys():
                self.progress["all_threads"]["thread"]["current_fold"] = 0

    def download_thread(self, current_thread):
        self.progress["all_threads"][current_thread] = {"folds":[], "current_fold": 0}
        path = self.thread_paths[current_thread]

        print('Downloading from {}...'.format(path))
        bashCommand = "wget " + path + " -P " + self.data_dir
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print('Reading and splitting file')
        threadpath = self.data_dir + path.split("/")[-1]
        reader = open(threadpath, 'r')
        docs = None
        if self.dataset == 'gigaword':
            docs = [el.split("\t") for el in reader.read().split("\n")]
            docs = [el[2] if len(el)==3 else "" for el in docs]
            docs = "".join(docs)
        else:
            docs = reader.read()
        return docs

    def make_folds(self, docs, current_thread, num_folds):
        split_size = math.ceil(len(docs) / num_folds)

        self.progress["all_threads"][current_thread]["folds"] = []
        for i in range(0, num_folds):
            this_fold = docs[split_size * i: split_size * (i+1)]
            foldpath = self.data_dir + "/fold{}.txt".format(i)
            foldfile = open(foldpath, 'w')
            foldfile.write(this_fold)

            self.progress["all_threads"][current_thread]["folds"].append({"finished":False, "path":foldpath})

            bashCommand = "rm " + self.data_dir + "/reviews.txt".format(current_thread + 1)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()


    def process(self, foldpath, progress):
        trainer = TrainLangModel()
        vocab = None
        params = None
        train_loss = None
        if progress["started"]:
            current = torch.load(self.savepath + '/model.pt')
            vocab = current["vocab"]
            params = current["state_dict"]
            train_loss = current["train_loss"]

        text = open(foldpath, 'r').read()
        trainer.current_loss = train_loss
        trainer.prepare_data(text, already_read = True, vocab = vocab)
        trainer.init_model(checkpoint_params = params)
        trainer.train_step(None, trainer.model, START)

        trainer.save_checkpoint(name = 'model.pt')




