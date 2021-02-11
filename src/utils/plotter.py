"""
	@author xxx xxx@cs.xxx.edu
	@date 08/14/2019
	Loading data
"""

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SMALL_SIZE = 8.5
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
plt.style.use('ggplot')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
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

	@staticmethod
	def plot_error(data, x, title, savepath):
		# matplotlib.rcParams['text.usetex'] = True
		fig = plt.figure(figsize=(7, 7))
		# r'\textbf{time (s)}'
		plt.plot(x, data[0, :], 'g:', label=r'ParM$_{MNIST}$')
		plt.plot(x, data[1, :], 'g', label=r'CI$_{MNIST}$')
		plt.plot(x, data[2, :], 'b:', label=r'ParM$_{Fashion}$')
		plt.plot(x, data[3, :], 'b', label=r'CI$_{Fashion}$')
		plt.plot(x, data[4, :], 'r:', label=r'ParM$_{CIFAR10}$')
		plt.plot(x, data[5, :], 'r', label=r'CI$_{CIFAR10}$')

		# plt.title("Relative Reconstruction Errors")
		plt.xlabel("K")
		plt.ylabel("Error (\%)")
		plt.legend(loc='upper left')

		plt.savefig(savepath, bbox__hnches='tight')
		plt.close(fig)

	@staticmethod
	def plot_bar(data, name, savepath):
		import seaborn as sns
		sns.set_theme(style="whitegrid")
		sns.set(font_scale = 3)
		g = sns.catplot(
		    data=data, kind="bar",
		    x="K", y="Accuracy", hue="Model",
		    ci="sd", palette="dark", alpha=.6, height=7
		)

		g.despine(left=True)
		if name == 'mnist':
			g.set_axis_labels("K", "Accuracy")
		else:
			g.set_axis_labels("K", "")

		if name == 'cifar10':
			g.legend.set_title('Model')
			# plt.setp(g._legend.get_title())
		else:
			g.legend.remove()
		# Show the plot
		plt.savefig(savepath, bbox__hnches='tight')
		plt.close()
		# plt.show()

	@staticmethod
	def plot_bar2(data, name, savepath):
		import seaborn as sns
		sns.set_theme(style="whitegrid")

		g = sns.catplot(
		    data=data, kind="bar",
		    x="K", y="Accuracy", hue="Class",
		    ci="sd", palette="bright", alpha=.6, height=5
		)

		g.despine(left=True)
		g.set_axis_labels("K", "Accuracy")
		g.legend.set_title('Class')
		# Show the plot
		plt.savefig(savepath, bbox__hnches='tight')
		plt.close()
		# plt.show()

	@staticmethod
	def plot_lines(data, name, savepath):
		import seaborn as sns
		sns.set_theme(style="whitegrid")

		sns.lineplot(data=flights, x="year", y="passengers")

		g = sns.catplot(
		    data=data, kind="bar",
		    x="K", y="Accuracy", hue="Model",
		    ci="sd", palette="dark", alpha=.6, height=5
		)

		g.despine(left=True)
		if name == 'mnist':
			g.set_axis_labels("K", "Accuracy")
		else:
			g.set_axis_labels("K", "")

		if name == 'cifar10':
			g.legend.set_title('Model')
		else:
			g.legend.remove()
		# Show the plot
		plt.savefig(savepath, bbox__hnches='tight')
		plt.close()


if __name__ == '__main__':
	import pandas
	# x = ['A', 2, 4, 10]
	# data = pandas.read_csv('../../results/rerrors.csv')
	# data = data.values
	# Plotter.plot_error(data, x, 'MNIST', '../../results/rerrors.png')
	names = ['mnist', 'fashion', 'cifar10']
	# names = ['cifar10']
	for name in names:
		data = pandas.read_csv('../../results/rerror_{}.csv'.format(name))
		Plotter.plot_bar(data, name, '../../results/accuracy_{}.png'.format(name))

	# name = 'multi'
	# data = pandas.read_csv('../../results/rerror_{}.csv'.format(name))
	# Plotter.plot_bar2(data, name, '../../results/accuracy_{}.png'.format(name))
