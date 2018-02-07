from time import time
START = time()

from pretraining.langmodel.trainer import TrainLangModel
import subprocess
import os
import json
import math
import torch
from torchtext import data
import argparse
from multiprocessing import Pool
import psutil
from collections import Counter


class Sequence:
    def __init__(self,
                timelimit = 3.75 * 60 * 60,
                max_size = 1000000,
                dataset = 'gigaword',
                vectors = 'glove',
                sources = ['nyt', 'wpb', 'ltw', 'apw']
            ):
        current_path = os.getcwd()
        self.timelimit = timelimit
        self.max_size = max_size
        self.vectors = vectors
        self.sources = sources

        if dataset == 'test':
            self.data_dir = 'data/gigaword'
            path = "https://s3.amazonaws.com/gigaword/gigaword_cleaned_{}.txt"
            self.thread_paths = [path.format(fold) for fold in ['train', 'val', 'test']]
        elif dataset == 'gigaword':
            self.data_dir = 'data/gigaword'
            path = "https://s3.amazonaws.com/gigaword/thread{}.txt"
            total_files = 15
            self.thread_paths = [path.format(i+1) for i in range(3, total_files)]
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

    def run(self, timelimit = False):
        current_thread = self.progress["current_thread"]

        if timelimit:
            already_downloaded = current_thread < len(self.progress["all_threads"])
            if not already_downloaded:

                assert current_thread == len(self.progress["all_threads"])
                self.progress["all_threads"].append({"folds":[], "current_fold": 0})
                path = self.thread_paths[current_thread]
                docs, size = self.download(path)
                self.split(docs, current_thread, size)

            self.run_fold(current_thread, timelimit)
        else:
            for path in self.thread_paths:
                self.progress["all_threads"].append(path)
                docs = self.download(path)
                self.process(docs, self.progress, already_read = True)

        '''
        if self.progress["current_thread"] >= len(self.thread_paths):
            print("FINISHED ITERATION, REINITING CYCLE")
            self.progress["iterations"] += 1
            self.progress["current_thread"] = 0
            for i in range(len(self.progress["all_threads"])):
                self.progress["all_threads"][i]["current_fold"] = 0
        '''

    def download(self, path, generate = None):
        print('Downloading from {}...'.format(path))
        bashCommand = "wget " + path + " -P " + self.data_dir
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print('Reading file...')
        threadpath = self.data_dir + "/" + path.split("/")[-1]
        reader = open(threadpath, 'r')
        size = os.path.getsize(threadpath)
        docs = None
        print(generate)
        if generate is not None:
            num_folds = math.ceil(size/generate)
            return self.get_folds(reader, generate)
        else:
            print('reading doc...')
            if self.dataset == 'gigaword':
                data = reader.read()
                docs = [el.split("\t") for el in data.split("\n")]
                docs = [el[2] for el in docs if len(el) == 3]
                docs = " ".join(docs)
            else:
                docs = reader.read()
            print('done')
            self.delete(threadpath)
            return docs
    

    def get_folds(self, reader, generate):
        if self.dataset == 'gigaword':
    	    while True:
    	        data = reader.read(generate)
    	        if not data:
    	    	     break
    	        docs = [el.split("\t") for el in data.split("\n")]
    	        docs = [el[2] if (len(el)==3) else "" for el in docs]
    	        docs = "".join(docs)
    	        yield docs

        elif self.dataset == 'test':
    	    while True:
    	        data = reader.read(generate)
    	        if not data:
    	    	    break
    	        yield data

    def split(self, docs, current_thread, size):
        print("file has size of {}".format(size))
        num_folds = math.ceil(size / self.max_size)
        print("Splitting into {} folds".format(num_folds))
        split_size = math.ceil(len(docs) / num_folds)

        self.progress["all_threads"][current_thread]["folds"] = []
        for i in range(0, num_folds):
            this_fold = docs[split_size * i: split_size * (i+1)]
            foldpath = self.data_dir + "/fold{}.txt".format(i)
            with open(foldpath, 'w') as f:
                f.write(this_fold)

            self.progress["all_threads"][current_thread]["folds"].append({"path":foldpath})

    def delete(self, threadpath):

        print("Deleting {}".format(threadpath))

        bashCommand = "rm " + threadpath
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    def get_datasets(self):
        print('Building vocab for full dataset...')

        for address in self.thread_paths:
            examples = [''] * 3000
            text = self.download(address, generate = 100000000)
            field = data.Field()
            fields = [('text', field)]
            for i, fold in enumerate(text):
                print('Reading fold:{}'.format(i))
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
                print('memory use:', memoryUse)
                examples[i] = data.Example.fromlist([fold], fields)
            examples = [ex for ex in examples if ex != '']
            dataset = data.Dataset(examples, fields)
            field.build_vocab(dataset)
            yield field.vocab.freqs

    def build_vocab(self):
        datasets = self.get_datasets()
        counter = Counter()
        for i, d in enumerate(datasets):
            counter = counter + d

        torch.save(counter, '{}_vocab.pt'.format(self.dataset))
        print('should be saved')

    def process(self, fold, progress, already_read = False):
        trainer = TrainLangModel(
                    tune_wordvecs = True,
                    wordvec_source = self.vectors,
                    time_limit = self.timelimit,
                    savepath = self.savepath,
                    max_vocab = 100000
                )

        params = None
        train_loss = None
        if os.path.exists(self.savepath + 'model.pt'):
            current = torch.load(self.savepath + 'model.pt')
            params = current["state_dict"]
            train_loss = current["train_loss"]

        counter = torch.load('vocab.pt')

        if train_loss is not None:
            trainer.current_loss = train_loss

        max_size = (2**29-1)
        num_folds = (len(fold) // max_size) + 1
        for i in range(num_folds):
            print(i)
            text = fold[max_size*i:max_size*(i+1)]
            if not already_read:
                with open(fold, 'r') as socket:
                    text = socket.read()
                    print("VOCAB")
                    print(vocab)
                    trainer.prepare_data(text, already_read = True, vocab = counter)
            else:
                print('fold:{}'.format(i))
                print('size:{}'.format(max_size))
                trainer.prepare_data(text, already_read = True, vocab = counter)

            trainer.init_model(checkpoint_params = params)
            trainer.train_step(None, trainer.model, START)

            trainer.save_checkpoint(name = 'model.pt')
        del trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
    parser.add_argument('--data',  type=str, default = 'reviews',
                        help='location of pretrained init')
    parser.add_argument('--timelimit',  type=int, default = None,
                        help='location of pretrained init')
    parser.add_argument('--max_size',  type=int, default = 1000000000,
                        help='location of pretrained init')
    parser.add_argument('--vectors',  type=str, default = '',
                        help='location of pretrained init')
    ARGS = parser.parse_args()
    Sequence(dataset = ARGS.data, vectors = ARGS.vectors, timelimit = ARGS.timelimit, max_size = ARGS.max_size)

