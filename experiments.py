import os
project_path = os.getcwd()

from classifier.attention_rnn.trainer import TrainClassifier
from pretraining.langmodel.trainer import TrainLangModel

attn_savepath = os.path.join(project_path, 'learned_attns')

MULTIPLE_DIMS = ['GloVe']

vector_cache = os.path.join(project_path, 'vectors')

labeled_datasets = ['IMDB']


#UNSUPERVISED PRETRAINING EXPERIMENTS HERE
unlabeled_datasets = ["wikitext"]
benchmark_dataset = ''

for dataset in unlabeled_datasets:
    savepath = os.path.join(project_path, 'trained_models', 'langmodel', dataset)
    os.makedirs(savepath, exist_ok = True)

    #TRAIN LANGUAGE MODEL ON UNLABALED DATA AND SAVE WEIGHTS
    model = TrainLangModel(
                data = dataset,
                savepath = savepath
            )

    model.train()
    model.evaluate()
    model.save_model(os.path.join(model.savepath, 'model.pt'))

    #INITIALIZE CLASSIFER WITH LANGUAGE MODEL WEIGHTS AND TRAIN
    classifier = TrainClassifier(
                    num_classes = labeled_datasets[dataset]['num_classes'],
                    pretrained_modelpath = savepath,
                    datapath = benchmark_dataset
                )
    classifier.train()
    loss = classifier.evaluate()
    print('AFTER UNSUPERVISED PRETRAINING ON THE {} DATASET,\nWE GET LOSS OF {}'.format(dataset, loss))


    attn_savepath = os.path.join(project_path, 'langmodel', dataset)
    os.makedirs(attn_savepath, exist_ok = True)

    classifier.dump_attns(os.path.join(attn_savepath, 'saved_attn_weights.pt'))


#VECTOR EXPERIMENTS HERE
vector_sources = ['GloVe', 'CharLevel', 'googlenews']
vector_dims = {el: [None] for el in vector_sources}
vector_dims['GloVe'] = ['300']

vector_savepath = os.path.join(attn_savepath, 'vectors')

for dataset in labeled_datasets:
    for source in vector_sources:
        for dim in vector_dims[source]:
            model = TrainClassifier(
                        pretrained_modelpath = None,
                        datapath = dataset,
                        wordvec_dim = dim,
                        wordvec_source = [source],
                        vector_cache = vector_cache
                    )
            model.train()
            loss = model.evaluate()
            print('WITH {} VECTORS WITH {} DIMS ON THE {} DATASET,\nWE GET LOSS OF {}'.format(source, dim, dataset, loss))

            dirpath = os.path.join(vector_savepath, source + dim + "d", dataset)
            os.makedirs(dirpath, exist_ok = True)
            model.dump_attns(os.path.join(dirpath, 'saved_attn_weights.pt'))




















