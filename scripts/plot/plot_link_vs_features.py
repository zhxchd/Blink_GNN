import argparse
import numpy as np
from matplotlib import pyplot as plt
import plot_utils
import matplotlib as mpl

parser = argparse.ArgumentParser(description='Plot MAE experiments.')
parser.add_argument("style", type=str, help="One of acmconf or old.")
args = parser.parse_args()

style = args.style

plot_utils.load_style(style)

mpl.rcParams.update({'figure.figsize': (2.1,1.5)})

gcn = {"cora": 0.8682422451994091, "citeseer": 0.7826923076923077, "lastfm": 0.8311997201818817, 'facebook': 0.8973003441319568}
mlp = {"cora": 0.710388970950271, "citeseer": 0.7368990384615386, "lastfm": 0.7003497726477789, 'facebook': 0.7647858075234365}
res = {'cora': 0.8504677498769077, 'citeseer': 0.7132211538461539, 'lastfm': 0.8600384749912557, 'facebook': 0.9345615284205528}

# labels = ["Facebook", "LastFM", "CiteSeer", "Cora"]
labels = ["Cora", "CiteSeer", "LastFM", "Facebook"]
fig, ax = plt.subplots()
x = np.arange(len(labels))
width = 0.4
ax.bar(x-width/2, [i*100 for i in mlp.values()], width=width, label="MLP (without links)", color="white", edgecolor="black", hatch="\\\\", linewidth=0.5)
# ax.bar(x, [i * 100 for i in res.values()], width=width, label="GCN (without features)", color="white", edgecolor="black", hatch="////", linewidth=0.5)
ax.bar(x+width/2, [i * 100 for i in gcn.values()], width=width, label="GCN", color="white", edgecolor="black", hatch="....", linewidth=0.5)
ax.set_xticks(x)
ax.tick_params(axis='x', which='both', width=0)
ax.set_xticklabels(labels)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylim(0,100)
ax.set_ylabel("Average accuracy (\%)")
# plt.show()
# ax.legend(ncol=1)
fig.savefig("figures/motivation.pdf", bbox_inches='tight')

fig_leg = plt.figure()
ax_leg = fig_leg.add_subplot()
ax_leg.axis('off')
legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=3)
fig = legend.figure
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("figures/motivation_legend.pdf", bbox_inches=bbox)