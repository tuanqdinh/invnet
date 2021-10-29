"""
	@author xxx xxx@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas


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

class Plotter:
	@staticmethod
	def plot_curves(data, x, off, title, ylabel, savepath):
		data =[[0.11267099 0.07208917 0.0632865  0.0609888  0.05900814 0.0565367 0.05452069 0.0539086  0.0522978  0.05095612]
		 [0.11374548 0.07706294 0.07021692 0.0689388  0.0676743  0.06597095  0.06473865 0.06461731 0.06357495 0.06284263]
		 [0.11259246 0.07631384 0.06962023 0.06833202 0.06715021 0.06551486  0.0642413  0.064136   0.06310341 0.06233186]]
		[[0.9657372  0.98613073 0.99221867 0.99437957 0.99466007 0.99449385
		  0.99493019 0.99455618 0.9956678  0.99577169]
		 [0.96444039 0.98165053 0.99072973 0.99222288 0.99193274 0.99053159
		  0.99196105 0.99105525 0.99287392 0.99271116]
		 [0.96740274 0.98024554 0.98985969 0.99363042 0.99245855 0.99064094
		  0.99284917 0.99232302 0.99318001 0.99220344]]

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

if __name__ == '__main__':
	names = ['mnist', 'fashion', 'cifar10']
	# names = ['cifar10']
	for name in names:
		data = pandas.read_csv('rerror_{}.csv'.format(name))
		Plotter.plot_bar(data, name, 'accuracy_{}.png'.format(name))
