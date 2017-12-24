#Visualize function taken from https://github.com/NVIDIA/sentiment-discovery
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_neuron_heatmap(text, values, savename=None, negate=False, cell_height=.325, cell_width=.15):
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
	num_rows = len(values)
	plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
	hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',
					xticklabels=False, yticklabels=False, cbar=False)
	plt.tight_layout()
	if savename is not None:
		plt.savefig(savename)
	# clear plot for next graph since we returned `hmap`
	plt.clf()
	return hmap
