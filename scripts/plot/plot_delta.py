import json
from matplotlib import pyplot as plt
import argparse
import plot_utils

parser = argparse.ArgumentParser(description='Plot MAE experiments.')
parser.add_argument("style", type=str, help="One of acmconf or old.")
args = parser.parse_args()

style = args.style

plot_utils.load_style(style)

# there will be 3 figures
with open("../output/delta.json") as f:
    delta = json.load(f)

fig = plt.figure()

x = ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]

plot_utils.plot(x, [delta["soft"]["citeseer"]["gcn"]["1"][i][0] for i in x], [delta["soft"]["citeseer"]["gcn"]["1"][i][1] for i in x], color="green", label=r"$\epsilon=1$")
plot_utils.plot(x, [delta["soft"]["citeseer"]["gcn"]["8"][i][0] for i in x], [delta["soft"]["citeseer"]["gcn"]["8"][i][1] for i in x], color="red", label=r"$\epsilon=8$")

plt.xticks(["0.1", "0.3", "0.5", "0.7", "0.9"])
plt.xlabel("$\delta$")
plt.ylabel("Accuracy (\%)")
# plt.legend()
plt.savefig(f"figures/delta.pdf", bbox_inches='tight')
legend=plt.legend()
ax = plt.gca()
# there will be figures popping up
plt.close(fig)

fig_leg = plt.figure()
ax_leg = fig_leg.add_subplot()
ax_leg.axis('off')
legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=2)
fig = legend.figure
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("figures/delta_legend.pdf", bbox_inches=bbox)