import os

project_path = os.getcwd()

import tinys3

keypath = 's3key.csv'


from pretraining.langmodel.trainer import TrainLangModel

vector_cache = os.path.join(project_path, 'vectors')

unlabeled_datasets = ["gigaword", "wikitext", "ptb"]
benchmark_dataset = ''

for dataset in unlabeled_datasets:
    savepath = os.path.join(project_path, 'trained_models', 'langmodel', dataset)
    os.makedirs(savepath, exist_ok = True)

    #TRAIN LANGUAGE MODEL ON UNLABALED DATA AND SAVE WEIGHTS
    model = TrainLangModel(
                data = dataset,
                savepath = savepath,
                num_epochs = 100
            )

    model.train()
    model.save_checkpoint('pretrained_model.pt')

    print('Connecting to s3.')
    keys = open(keypath, 'r').read().split('\n')

    access = keys[0][15:]
    secret = keys[1][13:]
    conn = tinys3.Connection(access, secret)
    print('Connected with S3.')

    print('uploading...')
    f = open(model.savepath + 'pretrained_model.pt', 'rb')
    conn.upload(dataset + '22', f, 'pretrainedlangmodels')
    print('Done')

    os.remove(model.savepath + 'pretrained_model.pt')

