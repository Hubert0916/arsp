from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent
res = json.loads((root / 'results_proxy.json').read_text())
budgets = np.asarray(res['budgets'], dtype=float)
x = np.log2(budgets + 1.0)
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.8,
})
fig, ax = plt.subplots(figsize=(3.25, 2.05))
ax.plot(x, res['asrp_brier'], marker='o', linewidth=1.45, markersize=3.8,
        color='#1f5a9d', label='ASRP')
ax.plot(x, res['empirical_mean_brier'], marker='s', linewidth=1.45, markersize=3.7,
        color='#b35c1e', label='Empirical Mean')
ax.set_xticks(x, [str(int(b)) for b in budgets])
ax.set_xlabel('Label budget, $n$', labelpad=2)
ax.set_ylabel('Brier score', labelpad=2)
values = np.r_[res['asrp_brier'], res['empirical_mean_brier']]
ax.set_ylim(float(values.min()) - .01, float(values.max()) + .015)
ax.grid(axis='y', color='#d9d9d9', linewidth=.55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(width=.7, length=3, pad=2)
ax.legend(frameon=False, loc='upper right', handlelength=2.0,
          borderaxespad=.2, labelspacing=.25)
fig.tight_layout(pad=.35)
fig.savefig(root / 'figures/public_proxy_curve.pdf', bbox_inches='tight')
