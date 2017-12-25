import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def get_heatmap(words, wordvalues, ax = None, savename=None, negate=False, cell_height=.325, cell_width=.15):
    assert len(words) == len(wordvalues)
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
    num_rows = len(values) + 1
    hmap = None
    if ax is None:
        plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
        hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',xticklabels=False, yticklabels=False, cbar=False)
        plt.tight_layout()
    else:
        hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',xticklabels=False, yticklabels=False, cbar=False, ax = ax)

    if savename is not None:
        plt.savefig(savename)
    return hmap

#LIST OF SENTENCES (LIST OF WORDS), LIST OF LIST OF VALS
def plot_attn(texts, vals, savepath = None):
    fig, axes= plt.subplots(len(texts))
    for i in range(len(sentences)):
        get_heatmap(datatext[i], datavals[i], ax = axes[i])
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)

if __name__ == '__main__':

    text = 'This is a longer sentence to get '
    two = 'a better idea! of how it deals? with DIFFERNT STuff'
    sentences = 2* [text, two, text, two]

    datatext, datavals = [], []
    for text in sentences:
        words = text.split(" ")
        datatext.append(words)
        vals = [(i - (len(words) / 2))/(len(words)/2) for i in range(len(words))]
        datavals.append(vals)
    plot_attn(datatext, datavals)

    '''
    fig.axes.append(ax1)
    fig.axes.append(ax2)
    plt.show()
    '''

