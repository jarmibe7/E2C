import matplotlib.pyplot as plt
import matplotlib as mpl
import json 

# Configure Matplotlib to use LaTeX for text rendering
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42  # Ensures TrueType fonts are used
plt.rcParams['ps.fonttype'] = 42   # Ensures TrueType fonts for PostScript output

# Set up bold face for figure generation
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.titlesize'] = 25
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 20
# mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlepad'] = 5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 3
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['xtick.major.pad'] = 3
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 3
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['ytick.major.pad'] = 3
mpl.rcParams['figure.subplot.hspace'] = 0.85
mpl.rcParams["axes.labelpad"] = 5

# Define box plot properties
bw_props = dict(linestyle='-', linewidth=2.5, color='black')
boxplot_kwargs = dict(
    vert=True,
    patch_artist=True,
    widths=0.5,
    whis=[0, 100],
    showfliers=False,
    medianprops=bw_props,
    boxprops=bw_props,
    capprops=bw_props,
    whiskerprops=bw_props,
    flierprops=dict(linestyle='', markersize=10, markeredgewidth=2, color='black'),
    showmeans=False,
    meanline=True
)


import json
import matplotlib.pyplot as plt
import numpy as np

# import data as json, this can be numpy or array too
with open(f"/home/ayush/Desktop/tutorials/rl_latent/E2C/src/data_gen/contact/contacts_seed_20.json", "r") as f:
    data = json.load(f)

environments = ['door', 'coffee', 'drawer', 'button', 'faucet']
objectives = ['pixel', 'dynamics', 'random']
objective_labels = {'random': 'Random', 'pixel': 'Pixel', 'dynamics': 'Dynamics'}
colors = {'random': '#ff8080', 'pixel': '#304674', 'dynamics': '#957dadff'}

# Prepare data for plotting
plot_data = {env: {obj: data[env][obj] for obj in objectives} for env in environments}

fig, axes = plt.subplots(1, len(environments), figsize=(20, 6), sharey=True)

for idx, env in enumerate(environments):
    ax = axes[idx]
    parts = []
    for j, obj in enumerate(objectives):
        values = plot_data[env][obj]
        # Violin plot for each objective
        # vp = ax.violinplot(
        #     values,
        #     positions=[j],
        #     showmeans=False,
        #     showmedians=True,
        #     showextrema=True,
        #     widths=0.8,
        #     bw_method=0.3, points=500
        # )
        # for pc in vp['bodies']:
        #     pc.set_facecolor(colors[obj])
        #     pc.set_alpha(0.7)
        # for partname in ('cbars', 'cmedians', 'cmaxes', 'cmins'):
        #     vp[partname].set_edgecolor('black')
        #     vp[partname].set_linewidth(2)
        # parts.append(vp)

        # boxplot for each objective
        bp = ax.boxplot(values,
                        vert=True, 
                        patch_artist=True, 
                        medianprops=bw_props, 
                        boxprops=dict(linestyle='-', linewidth=2.5, color='black'), 
                        capprops=bw_props, 
                        whiskerprops=bw_props, 
                        showfliers=False,
                        flierprops=dict(linestyle='', markersize=10, markeredgewidth=2, color='black'),
                        widths=0.5,
                        whis=[0, 100],
                        positions=[j],
                        showmeans=False,
                        # meanprops=mean_props,
                        meanline=True)

        for patch in bp["boxes"]:
            patch.set_facecolor(colors[obj])
            patch.set_alpha(0.7)
        ax.scatter(
            j * np.ones_like(values),
            values,
            alpha=0.5,
            color='black',
            s=40
        )
    ax.set_xticks(range(len(objectives)))
    ax.set_xticklabels([objective_labels[obj] for obj in objectives], rotation=20)
    ax.set_title(env.capitalize())
    if idx == 0:
        ax.set_ylabel("Contacts")
    ax.grid(True, axis='y', alpha=0.3)

# Add a legend
# from matplotlib.patches import Patch
# legend_handles = [Patch(facecolor=colors[obj], label=objective_labels[obj], alpha=0.7) for obj in objectives]
# fig.legend(handles=legend_handles, loc='upper right') #, fontsize=14)

plt.suptitle("Contacts per Objective Function Across Environments") #, fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()