import json
from matplotlib import pyplot as plt
import argparse
import plot_utils

parser = argparse.ArgumentParser(description='Plot MAE experiments.')
parser.add_argument("style", type=str, help="One of acmconf or old.")
args = parser.parse_args()

style = args.style

plot_utils.load_style(style)

# there will be 9 figures
with open("../output/results.json") as f:
    blink_res = json.load(f)
with open("../output/bl_results.json") as f:
    bl_res = json.load(f)

for dataset in ["cora", "citeseer", "lastfm", "facebook"]:
    for model in ["gcn", "gat", "graphsage"]:

        fig = plt.figure()

        x = ["1","2","3","4","5","6","7","8"]

        # draw non private accuracy (upper bound)
        plt.plot(x, [blink_res["hard"][dataset][model]["None"][0]*100 for i in x], linestyle=':', marker=" ", color="#b2df8a", label="$\epsilon=\infty$ (non-private)")

        # draw pure privacy MLP (lower bound)
        plt.plot(x, [blink_res["hard"][dataset]["mlp"]["None"][0]*100 for i in x], linestyle=':', marker=" ", color="#fb9a99", label="$\epsilon=0$ (MLP)")

        plot_utils.plot(x, [blink_res["hard"][dataset][model][i][0]*100 for i in x], [blink_res["hard"][dataset][model][i][1]*100 for i in x], color="b", label="Blink-Hard (ours)", fill=False)
        if model != "gat":
            plot_utils.plot(x, [blink_res["soft"][dataset][model][i][0]*100 for i in x], [blink_res["soft"][dataset][model][i][1]*100 for i in x], color="r", label="Blink-Soft (ours)", fill=False)
        plot_utils.plot(x, [blink_res["hybrid"][dataset][model][i][0]*100 for i in x], [blink_res["hybrid"][dataset][model][i][1]*100 for i in x], color="g", label="Blink-Hybrid (ours)", fill=False)
        plot_utils.plot(x, [bl_res[dataset][model]["rr"][i][0]*100 for i in x], [bl_res[dataset][model]["rr"][i][1]*100 for i in x], color="#dfc27d", label="RR",linestyle="dashed", fill=False)
        plot_utils.plot(x, [bl_res[dataset][model]["ldpgcn"][i][0]*100 for i in x], [bl_res[dataset][model]["ldpgcn"][i][1]*100 for i in x], color="#fb9a99", label="L-DPGCN", linestyle="dashed", fill=False)
        if model != "gat":
            plot_utils.plot(x, [bl_res[dataset][model]["solitude"][i][0]*100 for i in x], [bl_res[dataset][model]["rr"][i][1]*100 for i in x], color="#cab2d6", label="Solitude",linestyle="dashed", fill=False)

        # plt.ylim(ymin=0.2, ymax=0.9)
        # if plt.gca().get_ylim()[0] > 60:
        #     plt.ylim(ymin=60)
        plt.xlabel("$\epsilon$")
        plt.ylabel("Accuracy (\%)")

        plt.savefig(f"figures/{dataset}_{model}.pdf", bbox_inches='tight')
        legend=plt.legend()
        ax = plt.gca()
        # there will be figures popping up
        plt.close(fig)

fig_leg = plt.figure()
ax_leg = fig_leg.add_subplot()
ax_leg.axis('off')
legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
fig = legend.figure
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("figures/legend.pdf", bbox_inches=bbox)