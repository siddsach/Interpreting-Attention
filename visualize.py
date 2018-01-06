import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
import argparse
import os
from scipy.spatial.distance import cosine
import re

def get_heatmap(words, wordvalues, ax = None, savename=None, negate=False):
    assert len(words) == len(wordvalues), 'got {} words and {} values'.format(len(words), len(wordvalues))
    text, values = [], []
    for i, word in enumerate(words):
        text += list(word) + [" "]
        values += (len(word) + 1) * [wordvalues[i]]

    n_limit = 74
    num_chars = len(text)
    total_chars = math.ceil(num_chars/float(n_limit))*n_limit
    mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
    text = np.array(text+[' ']*(total_chars-num_chars))
    values = np.array(values+[0]*(total_chars-num_chars))
    if negate:
        values *= -1

    values = values.reshape(-1, n_limit)
    text = text.reshape(-1, n_limit)
    mask = mask.reshape(-1, n_limit)
    hmap = None
    hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',xticklabels=False, yticklabels=False, cbar=False, ax = ax)

    if savename is not None:
        plt.savefig(savename)
    return hmap

#LIST OF SENTENCES (LIST OF WORDS), LIST OF LIST OF VALS
def plot_attn(texts, vals, labels, savepath = None):
    cell_height=.05
    cell_width=.15
    n_limit = 74
    num_sentences = len(texts)
    row_heights = [(cell_height * (len(' '.join(sent) + ' ') // n_limit + 1) + 0.03) for sent in texts]
    print("ROWHEIGHTS")
    print(row_heights)
    fig = plt.figure(figsize = (cell_width*n_limit, sum(row_heights) * 10))
    gs = gridspec.GridSpec(num_sentences, 1, height_ratios=row_heights)#, figsize=(cell_width*74, cell_height*sum(row_heights)))
    numrows = 0
    for i in range(0, num_sentences):
        axis = plt.subplot(gs[i])
        numrows += (len(vals[i]) % n_limit) + 1
        get_heatmap(texts[i], vals[i], ax = axis)
        axis.set_xlabel(labels[i], size = 'large', labelpad = 1.0, horizontalalignment = 'left')

    plt.tight_layout()

    if savepath is None:
        plt.show()
    else:
        print(' Writing {}'.format(savepath))
        plt.savefig(savepath)

def make_attn_vizs(paths, max_examples = 5):
    all_data = {}

    for path in paths:
        all_data[path] = {}
        all_data[path]['words'], all_data[path]['vals'], all_data[path]['labels'], all_data[path]['groundtruth'] = extract(path)

    examples_to_viz = choose_examples(all_data, max_examples)

    print('Found {} examples to build attention visualization'.format(len(examples_to_viz)))

    for i, ex in enumerate(examples_to_viz):
        words = [all_data[path]['words'][ex] for path in paths]
        vals = [all_data[path]['vals'][ex][:len(words[0])] for path in paths]
        words = [sent[1:len(sent)-1] for sent in words]
        vals = [sent[1:len(sent)-1] for sent in vals]
        vals = [[(1.0 / max(data)) * val for val in data] for data in vals]
        preds = [all_data[path]['labels'][ex] for path in paths]
        targets = [all_data[path]['groundtruth'][ex] for path in paths]
        labels, dataset = get_labels(paths, preds, targets)
        name = 'vizs/{}/{}th_different_{}th_example.png'.format(dataset, i, ex)
        plot_attn(words, vals, labels, savepath = name)

def get_labels(paths, preds, targets):
    labels = len(paths) * ['']
    dataset = None
    for i, path in enumerate(paths):
        fields = path.split("/")
        dataset = fields[2]
        vectors = re.search('(best)(.*)(model.pt)', fields[4]).group(2)
        label = "Vectors:{} Predicted:{} Actual:{}".format(vectors, preds[i], targets[i])
        labels[i] = label

    return labels, dataset




def extract(path = 'clf4attn.pt'):
    trained = torch.load(path, map_location=lambda storage, loc: storage)
    vocab = trained['vocab']
    labels = trained['labels']
    train_attns = trained['train_attns']

    numrows = len(train_attns['text'])
    datawords = []
    datavals = []
    datalabels = []
    dataground = []

    for row in range(numrows):
        words = [vocab.itos[int(i)] for i in train_attns['text'][row] if i!=0]
        if len(words) != 0:
            datawords.append(words)
            vals = [j for j in train_attns['attn'][row][:len(words)]]
            datavals.append(vals)
            preds = labels.itos[int(train_attns['preds'][row]) + 1]
            targets = labels.itos[int(train_attns['targets'][row]) + 1]
            print(preds == targets)
            datalabels.append(preds)
            dataground.append(targets)

    return datawords, datavals, datalabels, dataground

def choose_examples(all_data, max_examples):
    paths = list(all_data.keys())
    num_examples = len(all_data[paths[0]]['words'])
    all_pairs = []

    for path1 in range(len(paths)):
        for path2 in range(path1+1, len(paths)):
            all_pairs.append((path1, path2))

    examples_to_viz = dict()
    for i in range(num_examples):
        for path1, path2 in all_pairs:
            data1 = all_data[paths[path1]]
            data2 = all_data[paths[path2]]
            words1 = data1['words'][i]
            words2 = data2['words'][i]
            vals1 = np.array([data1['vals'][i]])[:, :len(words1)]
            vals2 = np.array([data2['vals'][i]])[:, :len(words2)]
            assert vals1.shape == vals2.shape
            labels1 = data1['labels'][i]
            labels2 = data2['labels'][i]
            if labels1 != labels2:#and len(words1) > 12:
                #COMPUTE SIMILARITY

                distance = cosine(vals1[0], vals2[0])

                examples_to_viz[i] = distance

    print("Found {} examples with different answers and enough words".format(len(examples_to_viz)))

    most_relevant = sorted(examples_to_viz.items(), key = lambda k: k[1], reverse = True)

    if max_examples is not None:
        choices = [el[0] for el in most_relevant[:max_examples]]
    else:
        choices = [el[0] for el in most_relevant]


    '''
    for entropies in pairwise_entropy.values():
        ranked_examples = sorted(range(len(entropies)), key = lambda i: entropies[i])
        most_different = ranked_examples[:examples_per_pair]
        examples_to_viz.update(most_different)
    '''

    return choices

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
    prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
    first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Tuning Hyperparameters')
    parser.add_argument('--model_folder', type=str, default=None,
                        help='location of the data corpus')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='location of the data corpus')
    args = parser.parse_args()
    if args.model_folder is not None:
        paths = [args.model_folder + "/" + path for path in os.listdir(args.model_folder)]
        make_attn_vizs(paths, max_examples = args.max_examples)

    else:
        text = 3 * 'This is a longer sentence to get '
        two = 'a better idea! of how it deals? with DIFFERNT STuff'
        three = 'Is this too many sentences'
        four = 5 * 'shit I hope this all fits'
        sentences = [text, two, three, four]

        datatext, datavals, datalabels = [], [], []
        for text in sentences:
            words = text.split(" ")
            datatext.append(words)
            vals = [(i - (len(words) / 2))/(len(words)/2) for i in range(len(words))]
            datalabels.append("Label")
            datavals.append(vals)
        plot_attn(datatext, datavals, datalabels, savepath = 'test.png')

    '''
    fig.axes.append(ax1)
    fig.axes.append(ax2)
    plt.show()
    '''

