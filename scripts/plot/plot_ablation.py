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

# there will be 3 figures
with open("../output/ablation.json") as f:
    res = json.load(f)

# there will be 3 figures
with open("../output/density.json") as f:
    density = json.load(f)

# cm = 1/2.54 
# mpl.rcParams.update({
#     "font.family": "serif",
#     "font.size": 8,
#     'figure.figsize': (4*cm, 3*cm),
#     'axes.linewidth': 0.5,
#     'xtick.major.pad':0,
#     'ytick.major.pad':0,
#     "text.usetex": True,
#     'text.latex.preview': True,
#     'text.latex.preamble': [
#         r"""
#         \usepackage{libertine}
#         \usepackage[libertine]{newtxmath}
#         """]
#     })

# rc("figure", figsize=(4,3))

fig = plt.figure()
# plt.gca().xaxis.set_tick_params(width=0.5, size=2)
# plt.gca().yaxis.set_tick_params(width=0.5, size=2)
dataset = "citeseer"
x = ["1","2","3","4","5","6","7","8"]
plt.plot(x, [res[dataset][i]["prior"][0] for i in x], color="blue")#, linewidth=1)
plt.xlabel("$\epsilon$")#, labelpad=0)
# plt.ylabel("MAE")
# plt.legend()
plt.savefig(f"figures/ablation_{dataset}_prior.pdf", bbox_inches='tight')

plt.close(fig)

fig = plt.figure()
plt.gca().xaxis.set_tick_params(width=0.5, size=2)
plt.gca().yaxis.set_tick_params(width=0.5, size=2)
dataset = "citeseer"
x = ["1","2","3","4","5","6","7","8"]
plt.plot(x, [res[dataset][i]["full"]["mae"][0] for i in x], color="black", label=r"Full \textsc{Blink}")#, linewidth=1)
plt.plot(x, [res[dataset][i]["prior"][0] for i in x], color="blue", label="Prior only")#, linewidth=1)
plt.plot(x, [res[dataset][i]["evidence"][0] for i in x], color="red", label="Evidence only")#, linewidth=1)
# plot(x, [res[dataset][i]["full"]["mae"][0] for i in x], [res[dataset][i]["full"]["mae"][1] for i in x], color="black", label=r"Full \textsc{Blink}", fill=False)
# plot(x, [res[dataset][i]["prior"][0] for i in x], [res[dataset][i]["prior"][1] for i in x], color="blue", linestyle=":", label="Prior only", fill=False)
# plot(x, [res[dataset][i]["evidence"][0] for i in x], [res[dataset][i]["evidence"][1] for i in x], color="red", linestyle=":", label="Evidence only", fill=False)

plt.yscale("logit")
plt.xlabel("$\epsilon$")#, labelpad=0)
# plt.ylabel("MAE")
# plt.legend()
plt.savefig(f"figures/ablation_{dataset}.pdf", bbox_inches='tight')

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
fig.savefig("figures/ablation_legend.pdf", bbox_inches=bbox)