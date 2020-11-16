import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import sys

from scipy.signal import savgol_filter

lw = 2
fs = 20
option = int(sys.argv[1])
plot_type = ["all", "all1", "T", "static"][option]

colormap = [
    "tab:blue",
    "tab:brown",
    "tab:pink",
    "tab:orange",
    "tab:green",
    "saddlebrown",
    "tab:red",
    "deeppink",
    "darkcyan",
    "slateblue",
    "tab:gray",
    "royalblue",
]
markermap = ["x", "s", "v", "o", "d", "x", "*"]
marker_args = dict(
        markerfacecolor = "None",
        markeredgewidth=5,
        ms=14)

if plot_type == "table":
    lw = 4.0
    fig = plt.figure(figsize=(6,5))
    sns.set(style="whitegrid")
    x1, y1 = a[:,3], a[:,1]
    x2, y2 = b[:,3], b[:,1]
    x3, y3 = c[:,3], c[:,1]
    #x1 = [1.] + x1
    #y1 = [151.9561] + y1
    #x2 = [1.] + x2
    #y2 = [126.0844] + y2
    #x1, y1, x2, y2 = map(np.array, [x1, y1, x2, y2])

    plt.plot(x1, y1/y1[0], color=colormap[0], marker=markermap[0], markeredgecolor=colormap[0], lw=lw, label="easy", **marker_args)
    plt.plot(x2, y2/y2[0], color=colormap[2], marker=markermap[2], markeredgecolor=colormap[2], lw=lw, label="medium", **marker_args)
    plt.plot(x3, y3/y3[0], color=colormap[4], marker=markermap[4], markeredgecolor=colormap[4], lw=lw, label="hard", **marker_args)
    plt.xticks([1.0, 0.8, 0.6, 0.4, 0.2])
    plt.yticks([1.0, 0.8, 0.6, 0.4])
    plt.xlim(1.05, 0.15)
    plt.ylim(0.4, 1.05)
    plt.xlabel("Command rate", fontsize=fs)
    plt.ylabel("Relative reward", fontsize=fs)

    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontsize(fs)
    for label in ax.get_yticklabels():
        label.set_fontsize(fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("test-static.pdf", format="pdf")
    import sys
    sys.exit(0)


if plot_type == "all":
    experiments = [
        "aqmix165+ctr4",
        "aiqmix165+ctr4",
        "aqmix+interval165+ctr4",
        "aqmix+full165+ctr8",
        "aqmix+coach+ctr4",
        "aqmix+coach+vi2+ctr4+l20.001",
    ]
    labels = ["A-QMIX", "AI-QMIX", "A-QMIX (periodic)", "A-QMIX (full)", "COPA w/o", "COPA"]
elif plot_type == "all1":
    experiments = [
        "aqmix+ctr8",
        "aqmix+full+ctr8",
        "aiqmix+ctr8",
        "aqmix+coach+ctr8",
        "aqmix+coach+vi1+ctr8+l10.0001",
        "aqmix+coach+vi1+ctr4+l10.0001",
        "aqmix+full+coach+vi1+ctr8+l10.0001",
        "aqmix+full+coach+vi1+ctr4+l10.0001",
    ]
    labels = experiments
elif plot_type == "T":
    experiments = [
        "aqmix+coach+vi2+ctr2+l20.001",
        "aqmix+coach+vi2+ctr4+l20.001",
        "aqmix+coach+vi2+ctr8+l20.001",
        "aqmix+coach+vi2+ctr12+l20.001",
        "aqmix+coach+vi2+ctr16+l20.001",
        "aqmix+coach+vi2+ctr20+l20.001",
        "aqmix+coach+vi2+ctr24+l20.001",
    ]
    labels = ["T=2", "T=4", "T=8", "T=12", "T=16", "T=20", "T=24"]

runs = [0,1,2,3,4]
color_dict = {
        "aqmix165+ctr4": 0,
        "aiqmix165+ctr4": 1,
        "aqmix+interval165+ctr4": 2,
        "aqmix+full165+ctr8": 3,
        "aqmix+coach+ctr4": 4,
        "aqmix+coach+vi2+ctr4+l20.001": 6,
        "aqmix+coach+vi2+ctr2+l20.001": 5,
        "aqmix+coach+vi2+ctr8+l20.001": 7,
        "aqmix+coach+vi2+ctr12+l20.001": 8,
        "aqmix+coach+vi2+ctr16+l20.001": 9,
        "aqmix+coach+vi2+ctr20+l20.001": 10,
        "aqmix+coach+vi2+ctr24+l20.001": 11,
}

env_id = "mpe"

def smooth(y, n=100):
    y = np.array(y)
    y_ = savgol_filter(y, 101, 3)
    return y_

fig = plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
ax1 = plt.subplot(111)
xx = None

ax1.axhline(74.43, color='k', linestyle='-', label="greedy expert", lw=lw)
ax1.axhline(-4.38, color='k', linestyle='--', label="random", lw=lw)
if plot_type == "T":
    ax1.axhline(59.07, color=colormap[color_dict["aqmix165+ctr4"]], linestyle='-', label="A-QMIX", lw=lw)
    ax1.axhline(103.05, color=colormap[color_dict["aqmix+full165+ctr8"]], linestyle='-', label="A-QMIX (full)", lw=lw)

for i, (exp, lb) in enumerate(zip(experiments, labels)):
    x = None; Y = []
    min_len = 1000000000
    for run in runs:
        filename = f"/home/liub/Desktop/mount/teamstrategy/coach1/{env_id}/{exp}/run{run}/logs/stats.npy"
        if not os.path.exists(filename):
            continue
        print(filename)
        try:
            data = np.load(filename, allow_pickle=True).item()
            x, y = zip(*data.get('reward'))
            Y.append(smooth(y))
        except:
            continue
        min_len = min(min_len, len(y))
    if x is None:
        continue
    x = np.array(x[:min_len]) / 1e6
    Y = np.array([y[:min_len] for y in Y])
    mu = Y.mean(0)
    std = Y.std(0)
    if "aqmix165" in exp:
        print("aqmix ", mu[-1])
    if "aqmix+full165" in exp:
        print("full ", mu[-1])

    ax1.plot(x, mu, color=colormap[color_dict[exp]], alpha=1.0, lw=lw, ls="-")
    ax1.fill_between(x, mu-std, mu+std, color=colormap[color_dict[exp]], alpha=0.2)

ax1.set_xlabel("Timestep (mil)", fontsize=fs)
ax1.set_ylabel("Reward", fontsize=fs)

for label in ax1.get_xticklabels():
    label.set_fontsize(fs)
for label in ax1.get_yticklabels():
    label.set_fontsize(fs)

plt.ylim(-10, 120)
plt.xlim(0, 5)
#plt.text(5.1, -23, "x 1e6", fontsize=12)
plt.tight_layout()
plt.savefig(f"iclrimgs/{plot_type}.pdf", format="pdf")
#plt.savefig(f"iclrimgs/{plot_type}.png")
plt.close()

import pylab 
fig = pylab.figure(facecolor='white')
legend_fig = pylab.figure(facecolor='white')

if plot_type == "all":
    fig.gca().plot(range(10), pylab.randn(10), ls='-', color="k", lw=lw, label="greedy expert")
    fig.gca().plot(range(10), pylab.randn(10), ls='--', color="k", lw=lw, label="random")
if plot_type == "T":
    fig.gca().plot(range(10), pylab.randn(10), ls='-', color=colormap[color_dict["aqmix165+ctr4"]], lw=lw, label="A-QMIX")
    fig.gca().plot(range(10), pylab.randn(10), ls='-', color=colormap[color_dict["aqmix+full165+ctr8"]], lw=lw, label="A-QMIX (full)")
for i, (l, ex) in enumerate(zip(labels, experiments)):
    fig.gca().plot(range(10), pylab.randn(10), color=colormap[color_dict[ex]], lw=lw, label=l)
legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(), loc='center')
legend.get_frame().set_linewidth(lw)
legend_fig.canvas.draw()
legend_fig.tight_layout()
legend_fig.savefig(f"iclrimgs/{plot_type}-legend.pdf", facecolor=fig.get_facecolor(), format="pdf", bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))
#legend_fig.savefig(f"iclrimgs/{plot_type}-legend.png", facecolor=fig.get_facecolor(), bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))
plt.close()
