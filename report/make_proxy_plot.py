"""Plot mean Random-acquisition Brier curves in a compact paper style."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent
res = json.loads((root / 'results_proxy.json').read_text())
x = np.log2(np.asarray(res['budgets'], dtype=float) + 1)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'axes.linewidth': .7,
    'pdf.fonttype': 42,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})
fig, ax = plt.subplots(figsize=(4.6, 2.15))
for key, label, color, marker, face, zorder in [
    ('asrp_random', 'ASRP', '#0072B2', 'o', '#0072B2', 3),
    ('empirical_mean_random', 'Empirical Mean', '#D55E00', 's', 'white', 2),
]:
    ax.plot(x, res['combinations'][key]['brier'], label=label, color=color,
            linewidth=1.35, marker=marker, markersize=4,
            markerfacecolor=face, markeredgecolor=color, markeredgewidth=.9,
            zorder=zorder)

ax.set_xticks(x, [str(n) for n in res['budgets']])
ax.set_xlabel(r'Label budget, $n$', labelpad=3)
ax.set_ylabel('Brier score', labelpad=4)
ax.set_xlim(-.15, 5.15)
ax.set_ylim(.16, .36)
ax.set_yticks([.16, .20, .24, .28, .32, .36])
ax.grid(axis='y', color='#e5e5e5', linewidth=.5)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(direction='out', width=.7, length=3, pad=3)
ax.legend(frameon=False, loc='upper right', handlelength=2,
          handletextpad=.6, borderaxespad=.3, labelspacing=.4)
fig.tight_layout(pad=.5)
fig.savefig(root / 'figures/public_proxy_curve.pdf', bbox_inches='tight', pad_inches=.025)
plt.close(fig)
