# plt.style.use('ggplot')

# pth = '../results/invnet_latency_10_5000_2.npy'
# x = np.load(pth, allow_pickle=True)
# x = x.item()
# latency = x['latency']
# latency = sorted(latency)
# y = latency[:4500]

# plt.hist(y, bins=10)
# plt.savefig('inv_latency.png')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import pandas

matplotlib.style.use('seaborn')
params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
plt.rcParams.update(params)

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=15)

names = ['cifar10']
full_names = ['cifar10']
labels = ['Median', 'Mean', '99th', '99.5th', '99.9th']

width = 0.25  # the width of the bars
x = width*1.75 + np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()

data 


# 10 of this class
rects1 = ax.bar(x - width * 1.25, latency[0], width, label='ParMs', color='gray')
rects2 = ax.bar(x - width * 0.25, latency[1], width, label='Ours', color='orange')
rects3 = ax.bar(x + width * 0.75, latency[2], width, label='w/o stragglers', color='olive')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Latency (ms)', fontsize=20)
# ax.get_yaxis().set_visible(False)
ax.set_ylim([0,150])
ax.set_xlim([0, len(labels) - width*1])
# ax.set_title(full_names[0], fontsize=10)
ax.set_xticks(x)
ax.grid(False)
ax.set_xticklabels(labels, ha='center')
ax.set_aspect(aspect=0.01)


fig.tight_layout()
width = 6.3
height = 4

fig.set_size_inches(width, height)

fig.savefig('../samples/latency__app_k{}.pdf'.format(k), bbox_inches='tight')