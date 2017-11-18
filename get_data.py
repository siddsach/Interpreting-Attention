from torchtext import datasets
from torchtext import data
from torchtext.vocab import GloVe, CharNGram


################################################
#GET VECTORS HERE
################################################

#GloVe
print('Downloading GloVe Vectors...')
glove = GloVe(name = '6B', cache = 'vectors')

#CharNGram
print('Downloading CharNGram Vectors')
charVec = CharNGram(cache = 'vectors')

################################################
#LANGUAGE MODELING DATASETS HERE###########
################################################


#WIKITEXT
sentence_field = data.Field(
                    sequential = True,
                    use_vocab = True,
                    init_token = '<BOS>',
                    eos_token = '<EOS>',
                    lower = True,
                    tokenize = 'spacy',
                )

target_field = data.Field(sequential = False, batch_first = True)

print("Downloading wikitext data...")
train_sentences, valid_sentences, test_sentences = datasets.WikiText2.splits(
                                                                sentence_field,
                                                                root = 'data',
                                                                train = 'wikitext-2/wiki.train.tokens',
                                                                validation = 'wikitext-2/wiki.valid.tokens',
                                                                test = 'wikitext-2/wiki.test.tokens'
                                                            )



################################################
#LABELED TEXT CLASSIFICATION DATASETS HERE #####
################################################

#IMDB
print('Downloading IMDB data...')
train_data, test_data = datasets.IMDB.splits(
                                text_field = sentence_field,
                                label_field = target_field,
                                root = 'data', test = None
                            )
print("Done.")
