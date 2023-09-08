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
with open("../output/density.json") as f:
    density = json.load(f)

actual_dens = {
    "cora": 10556,
    "citeseer": 9104,
    "lastfm": 55612,
    "facebook": 342004
}

for dataset in ["cora", "citeseer", "lastfm", "facebook"]:

    fig = plt.figure()

    x = ["1","2","3","4","5","6","7","8"]
    plt.plot(x, [actual_dens[dataset] for i in x], linestyle=':', marker=" ", color="#33a02c", label="$\Vert A\Vert_1$")
    plot_utils.plot(x, [density[dataset][i][0] for i in x], [density[dataset][i][1] for i in x], color="blue", label="$\Vert\hat A\Vert_1$", fill=False)
    plot_utils.plot(x, [density[dataset][i][2] for i in x], [density[dataset][i][3] for i in x], color="green", label="Num. of true positive entries in $\hat A$ against $A$", fill=False)
    plt.fill_between(x, [density[dataset][i][2] for i in x], color="#b2df8a")

    plt.yscale("log")
    plt.xlabel("$\epsilon$")
    plt.savefig(f"figures/density_{dataset}.pdf", bbox_inches='tight')
    legend=plt.legend()
    ax = plt.gca()
    # there will be figures popping up
    plt.close(fig)


fig_leg = plt.figure()
ax_leg = fig_leg.add_subplot()
ax_leg.axis('off')
legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=3)
fig = legend.figure
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("figures/density_legend.pdf", bbox_inches=bbox)