import argparse
import json
from matplotlib import pyplot as plt
import plot_utils

parser = argparse.ArgumentParser(description='Plot MAE experiments.')
parser.add_argument("style", type=str, help="One of acmconf or old.")
args = parser.parse_args()

style = args.style

plot_utils.load_style(style)

# there will be 3 figures
with open("../output/mae.json") as f:
    mae = json.load(f)

for dataset in ["cora", "citeseer", "lastfm", "facebook"]:

    fig = plt.figure()

    x = ["1","2","3","4","5","6","7","8"]
    plt.plot(x, [mae[dataset][i][2] for i in x], linestyle=':', marker=" ", color="green", label="MAE upper bound")
    plot_utils.plot(x, [mae[dataset][i][0] for i in x], [mae[dataset][i][1] for i in x], color="blue", label="Empirical MAE", fill=False)

    # plt.yscale("log")
    plt.xlabel("$\epsilon$")
    plt.savefig(f"figures/mae_{dataset}.pdf", bbox_inches='tight')
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
fig.savefig("figures/mae_legend.pdf", bbox_inches=bbox)