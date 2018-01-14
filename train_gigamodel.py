from time import time
START = time()

from pretraining.langmodel.trainer import TrainLangModel
import subprocess
import os
import json
import math
import torch
import argparse
from multiprocessing import Pool


class Sequence:
    def __init__(self,
                timelimit = 3.75 * 60 * 60,
                max_size = 1000000,
                dataset = 'gigaword',
                vectors = 'glove',
                sources = ['NYT', 'WPB', 'LTW', 'APW']
            ):
        current_path = os.getcwd()
        self.timelimit = timelimit
        self.max_size = max_size
        self.vectors = vectors
        self.sources = sources

        if dataset == 'gigaword':
            self.data_dir = 'data/gigaword'
            path = "https://s3.amazonaws.com/gigaword/thread{}.txt"
            total_files = 10
            self.thread_paths = [path.format(i+1) for i in range(total_files)]
        elif dataset == 'reviews':
            self.data_dir = 'data/reviews'
            total_files = 1
            self.thread_paths = ["https://s3.amazonaws.com/amazonmoviereviews/reviews.txt"]

        self.dataset = dataset
        self.savepath = current_path + '/trained_models/langmodel/{}/training/'.format(self.dataset)
        trained_path = 'trained_models/langmodel'
        progresspath = trained_path + '/{}/progress.json'.format(dataset)
        if os.path.exists(progresspath):
            print("RESUMING TRAINING")
            self.progress = json.load(open(progresspath, 'r'))
            self.run()
            self.progress["started"] = True
        else:
            print("SETTING UP FOR TRAIN")
            self.progress = {"all_threads":[], "current_thread":0, "iterations": 0, "started": False}
            self.build_vocab()
            json.dump(self.progress, open(progresspath, 'w'))



    def run_fold(self, current_thread):
        current_fold = self.progress["all_threads"][current_thread]["current_fold"]
        print("Running fold: {}".format(current_fold))
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

        already_downloaded = current_thread < len(self.progress["all_threads"])
        if not already_downloaded:

            assert current_thread == len(self.progress["all_threads"])
            self.progress["all_threads"].append({"folds":[], "current_fold": 0})
            path = self.thread_paths[current_thread]
            docs, size = self.download(path)
            self.split(docs, current_thread, size)

        self.run_fold(current_thread)

        if self.progress["current_thread"] >= len(self.thread_paths):
            print("FINISHED ITERATION, REINITING CYCLE")
            self.progress["iterations"] += 1
            self.progress["current_thread"] = 0
            for i in range(len(self.progress["all_threads"])):
                self.progress["all_threads"][i]["current_fold"] = 0

    def download(self, path):
        print('Downloading from {}...'.format(path))
        bashCommand = "wget " + path + " -P " + self.data_dir
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print('Reading file...')
        threadpath = self.data_dir + "/" + path.split("/")[-1]
        reader = open(threadpath, 'r')
        size = os.path.getsize(threadpath)
        docs = None
        if self.dataset == 'gigaword':
            docs = [el.split("\t") for el in reader.read().split("\n")]
            docs = [el[2] if (len(el)==3 and el[0][:3] in self.sources) else "" for el in docs]
            docs = "".join(docs)
        else:
            docs = reader.read()

        self.delete(threadpath)

        print('Done.')

        return docs, size

    def split(self, docs, current_thread, size):
        print("file has size of {}".format(size))
        num_folds = math.ceil(size / self.max_size)
        print("Splitting into {} folds".format(num_folds))
        split_size = math.ceil(len(docs) / num_folds)

        self.progress["all_threads"][current_thread]["folds"] = []
        for i in range(0, num_folds):
            this_fold = docs[split_size * i: split_size * (i+1)]
            foldpath = self.data_dir + "/fold{}.txt".format(i)
            foldfile = open(foldpath, 'w')
            foldfile.write(this_fold)

            self.progress["all_threads"][current_thread]["folds"].append({"path":foldpath})

    def delete(self, threadpath):

        print("Deleting {}".format(threadpath))

        bashCommand = "rm " + threadpath
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    def build_vocab(self):
        print('Building vocab for full dataset...')
        all_text = ""
        for address in self.thread_paths:
            text, size = self.download(address)
            all_text += text

        trainer = TrainLangModel(wordvec_source = self.vectors, savepath = self.savepath)
        trainer.load_data(all_text, True)
        trainer.get_vectors(vocab = None)
        trainer.init_model()
        print('here')
        trainer.save_checkpoint('vocab.pt')
        print('should be saved')

    def process(self, foldpath, progress):
        trainer = TrainLangModel(
                    tune_wordvecs = True,
                    wordvec_source = self.vectors,
                    time_limit = self.timelimit,
                    savepath = self.savepath
                )

        vocab = None
        params = None
        train_loss = None
        if progress["started"]:
            current = torch.load(self.savepath + 'model.pt')
            vocab = torch.load(self.savepath + 'vocab.pt')
            params = current["state_dict"]
            train_loss = current["train_loss"]

        text = open(foldpath, 'r').read()
        if train_loss is not None:
            trainer.current_loss = train_loss
        trainer.prepare_data(text, already_read = True, vocab = vocab)
        trainer.init_model(checkpoint_params = params)
        trainer.train_step(None, trainer.model, START)

        trainer.save_checkpoint(name = 'model.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
    parser.add_argument('--data',  type=str, default = 'reviews',
                        help='location of pretrained init')
    parser.add_argument('--timelimit',  type=int, default = 3.75*60*60,
                        help='location of pretrained init')
    parser.add_argument('--max_size',  type=int, default = 1000000000,
                        help='location of pretrained init')
    parser.add_argument('--vectors',  type=str, default = 'glove',
                        help='location of pretrained init')
    ARGS = parser.parse_args()
    Sequence(dataset = ARGS.data, vectors = ARGS.vectors, timelimit = ARGS.timelimit, max_size = ARGS.max_size)


