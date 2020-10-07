"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE


SMALL_SIZE = 8.5
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
plt.style.use('ggplot')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('text', usetex=True)

class Plotter:
	@staticmethod
	def plot_curves(data, x, off, title, ylabel, savepath):
		fig = plt.figure(figsize=(7, 7))
		plt.plot(x, data[:, 0, off], label='Train', c='b')
		plt.plot(x, data[:, 1, off], label='Validation', c='g')
		plt.plot(x, data[:, 2, off], label='Test', c='r')

		plt.title(title)
		plt.xlabel("epoch")
		plt.ylabel(ylabel)
		if off == 0:
			plt.legend(loc='upper right')
		else:
			plt.legend(loc='lower right')

		plt.savefig(savepath, bbox__hnches='tight')
		plt.close(fig)
