import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import sys

from scipy.signal import savgol_filter

lw = 2.0
fs = 20
option = int(sys.argv[1])
plot_type = ["all", "all1", "T", "static"][option]

colormap = ["tab:blue", "tab:brown", "tab:pink", "tab:orange", "tab:green", "tab:gray", "tab:red", "tab:purple", "tab:cyan", "tab:olive"]
markermap = ["x", "s", "v", "o", "d", "x", "*"]
marker_args = dict(
        markerfacecolor = "None",
        markeredgewidth=5,
        ms=14)

if plot_type == "static":
    lw = 4.0
    fig = plt.figure(figsize=(6,5))
    sns.set(style="whitegrid")
    a = np.load("static-A", allow_pickle=True)
    b = np.load("static-B", allow_pickle=True)
    c = np.load("static-C", allow_pickle=True)
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
        "aqmix+ctr8",
        "aqmix+full+ctr8",
        "aiqmix+ctr8",
        "aqmix+coach+ctr8",
        "aqmix+coach+vi1+ctr8+l10.0001",
    ]
    labels = ["A-QMIX", "A-QMIX (full)", "AI-QMIX", "CP-QMIX w/o L1", "CP-QMIX"]
elif plot_type == "all1":
    experiments = [
            #"aqmix+ctr8",
            "aqmix165+ctr4",
            #"aqmix+interval+ctr8",
            "aqmix+interval165+ctr4",
            #"aqmix+full+ctr8",
            "aqmix+full165+ctr8",
            #"aiqmix+ctr8",
            "aiqmix165+ctr4",
            "aqmix+coach+ctr4",
            #"aqmix+coach+vi1+ctr8+l10.001",
            #"aqmix+coach+vi1+ctr8+l15e-05",
            #"aqmix+coach+vi2+ctr8+l20.001+l30.002",
            #"aqmix+coach+vi2+ctr2+l20.001",
            "aqmix+coach+vi2+ctr4+l20.001",
            #"aqmix+coach+vi2+ctr8+l20.001",
            #"aqmix+coach+vi2+ctr12+l20.001",
            #"aqmix+coach+vi2+ctr16+l20.001",
            #"aqmix+coach+vi2+ctr20+l20.001",
            #"aqmix+coach+vi2+ctr24+l20.001",
            #"aqmix+coach+vi2+ctr8+l20.001",
            #"aqmix+coach+vi2+ctr8+l20.001+l30.002",
            #"aqmix+coach+vi2+ctr8+l20.0005+l30.001",
            #"aqmix+coach+vi2+ctr8+l20.004+l30.0001",
            #"aqmix+coach+vi2+ctr8+l20.0002+l30.001",
            #"aqmix+coach+vi2+ctr8+l20.0005+l30.0002",
            #"aqmix+coach+vi2+ctr8+l20.002",
            #"aqmix+coach+vi2+ctr8+l20.004",
            #"aqmix+coach+vi2+ctr8+l20.01",
            #"aqmix+coach+vi2+ctr8+l20.02",
    ]
    labels = experiments
elif plot_type == "T":
    experiments = [
        "aqmix+coach+vi1+ctr2+l10.0001",
        "aqmix+coach+vi1+ctr4+l10.0001",
        "aqmix+coach+vi1+ctr8+l10.0001",
        "aqmix+coach+vi1+ctr12+l10.0001",
        "aqmix+coach+vi1+ctr16+l10.0001",
        "aqmix+coach+vi1+ctr20+l10.0001",
        "aqmix+coach+vi1+ctr24+l10.0001",
    ]
    labels = ["T=2", "T=4", "T=8", "T=12", "T=16", "T=20", "T=24"]

runs = [0,1,2,3,4]
color_dict = {
        "aqmix+ctr8" : 0,
        "aqmix+full+ctr8": 1,
        "aiqmix+ctr8": 2,
        "aqmix+coach+ctr8": 3,
        "aqmix+coach+vi1+ctr8+l10.0001": 4,
        "aqmix+coach+vi1+ctr4+l10.0001": 5,
        "aqmix+coach+vi1+ctr2+l10.0001": 3,
        "aqmix+full+coach+vi1+ctr8+l10.0001": 6,
        "aqmix+full+coach+vi1+ctr4+l10.0001": 7,
        "aqmix+coach+vi1+ctr12+l10.0001": 6,
        "aqmix+coach+vi1+ctr16+l10.0001": 7,
        "aqmix+coach+vi1+ctr20+l10.0001": 8,
        "aqmix+coach+vi1+ctr24+l10.0001": 9,}
        #"aqmix+ctr8" : 0,
        #"aqmix+full+ctr8": 1,
        #"aiqmix+ctr8": 2,
        #"aqmix+coach+ctr8": 3,
        #"aqmix+coach+vi1+ctr8+l10.0001": 4,
        #"aqmix+coach+vi2+ctr8+l20.001": 5,
        #"aqmix+coach+vi1+vi2+ctr8+l10.0001+l20.001": 6,
        #"aqmix+coach+vi1+vi2+ctr4+l10.0001+l20.001": 7,
        #"aqmix+coach+vi1+vi2+ctr12+l10.0001+l20.001": 8,
        #"aqmix+coach+vi1+vi2+ctr16+l10.0001+l20.001": 9}
            

env_id = "mpe"

def smooth(y, n=100):
    y = np.array(y)
    y_ = savgol_filter(y, 101, 3)
    return y_

fig = plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
ax1 = plt.subplot(111)
xx = None

ax1.axhline(85.6, color='k', linestyle='-', label="greedy expert", lw=lw)
ax1.axhline(-3.2, color='k', linestyle='--', label="random", lw=lw)

for i, (exp, lb) in enumerate(zip(experiments, labels)):
#for i, exp in enumerate(os.listdir("/home/liub/Desktop/mount/teamstrategy/coach1/mpe/")):
    #if "vi" in exp and ("vi1" not in exp):
    #    continue
    #if not ("vi2" in exp or "coach+ctr8" in exp or "full" in exp):
#for i, exp in enumerate(os.listdir("/home/liub/Desktop/mount/teamstrategy/yfysc/mpe/")):
    lb = exp
    x = None; Y = []
    xx = None;
    min_len = 1000000000
    for run in runs:
        #if "interval" in exp:
        #    filename = f"/home/liub/Desktop/mount/teamstrategy/iclr_results3/{env_id}/{exp}/run{run}/logs/stats.npy"
        #else:
        #if "full" in exp or "vi2+vi3" in exp:
        #    filename = f"/home/liub/Desktop/mount/teamstrategy/diverge/{env_id}/{exp}/run{run}/logs/stats.npy"
        #else:
        #    filename = f"/home/liub/Desktop/mount/teamstrategy/tmp/{env_id}/{exp}/run{run}/logs/stats.npy"
        #if "0.001" in exp:
        #if "vi3" in exp and run > 0:
        #    continue
        #if "vi2" in exp or "vi3" in exp:
        #    filename = f"/home/liub/Desktop/mount/teamstrategy/larger/{env_id}/{exp}/run{run}/logs/stats.npy"
        #else:
        #    filename = f"/home/liub/Desktop/mount/teamstrategy/models/{env_id}/{exp}/run{run}/logs/stats.npy"
        #if "v" not in exp:
        #    filename = f"/home/liub/Desktop/mount/teamstrategy/models/{env_id}/{exp}/run{run}/logs/stats.npy"
        #else:
        filename = f"/home/liub/Desktop/mount/teamstrategy/coach1/{env_id}/{exp}/run{run}/logs/stats.npy"
        if not os.path.exists(filename):
            continue
        print(filename)
        try:
            data = np.load(filename, allow_pickle=True).item()
            x, y = zip(*data.get('reward'))
            Y.append(smooth(y))
            xx = x
            min_len = min(min_len, len(y))
        except:
            continue
    if xx is None:
        continue
    x = np.array(xx[:min_len]) / 1e6
    Y = np.array([y[:min_len] for y in Y])
    mu = Y.mean(0)
    #std = Y.std(0) / np.sqrt(Y.shape[0])
    std = Y.std(0)

    ax1.plot(x, mu, alpha=1.0, lw=lw, ls="-", label=lb)
    ax1.fill_between(x, mu-std, mu+std, alpha=0.2)

ax1.set_xlabel("Timestep (mil)", fontsize=fs)
ax1.set_ylabel("Reward", fontsize=fs)
ax1.legend(loc="lower right", fontsize=7)

for label in ax1.get_xticklabels():
    label.set_fontsize(fs)
for label in ax1.get_yticklabels():
    label.set_fontsize(fs)

plt.ylim(-15, 130)
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(f"iclrimgs/920.png", format="png")
plt.close()
