import matplotlib as mpl
from matplotlib import pyplot as plt
import warnings

def load_style(style):
    warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)
    plt.style.use(f'./{style}.mplstyle')

def plot(x, y, yerr, color, label, linestyle="solid", fill=False):
    plt.plot(x, y, marker=" ", ls=linestyle, color=color, label=label)
    plt.errorbar(x, y, yerr=yerr, fmt="none", color=color)
    if fill:
        plt.fill_between(x, [a_i - b_i for a_i, b_i in zip(y, yerr)], [a_i + b_i for a_i, b_i in zip(y, yerr)], alpha=0.25, color=color)