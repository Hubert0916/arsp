"""Plot Random-acquisition scores and paired predictor improvements."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).parent
res=json.loads((root/'results_proxy.json').read_text())
x=np.log2(np.asarray(res['budgets'],dtype=float)+1)
asrp=np.asarray(res['combinations']['asrp_random']['brier'])
baseline=np.asarray(res['combinations']['empirical_mean_random']['brier'])
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,
                     'ytick.labelsize':7.5,'legend.fontsize':7,'axes.linewidth':.8,
                     'pdf.fonttype':42})
fig,(score_ax,delta_ax)=plt.subplots(ncols=2,figsize=(6.4,2.15))
for values,name,color,marker in [(asrp,'ASRP','#1f5a9d','o'),
                                (baseline,'Empirical Mean','#b35c1e','s')]:
    score_ax.plot(x,values,marker=marker,linewidth=1.4,markersize=3.5,color=color,label=name)
score_ax.set_title('(a) Random acquisition',loc='left',fontsize=8.5,pad=7)
score_ax.set_ylabel('Brier score (lower is better)',labelpad=3)
score_ax.set_ylim(.155,.365);score_ax.set_yticks([.16,.20,.24,.28,.32,.36])
score_ax.legend(frameon=False,loc='upper right',handlelength=1.5,borderaxespad=.1,labelspacing=.3)
difference=baseline-asrp
bars=delta_ax.bar(x,difference,width=.55,color='#1f5a9d')
delta_ax.bar_label(bars,labels=[f'{v:.3f}' for v in difference],padding=3,fontsize=7)
delta_ax.set_title('(b) Brier reduction',loc='left',fontsize=8.5,pad=7)
delta_ax.set_ylabel('Empirical Mean − ASRP',labelpad=3)
delta_ax.set_ylim(min(0,float(difference.min())-.01),max(.01,float(difference.max())*1.22))
delta_ax.axhline(0,color='#454545',linewidth=.8)
for ax in (score_ax,delta_ax):
    ax.set_xticks(x,[str(n) for n in res['budgets']]);ax.set_xlabel('Label budget, $n$',labelpad=2)
    ax.grid(axis='y',color='#e0e0e0',linewidth=.55);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.7,length=3,pad=2)
    ax.set_xlim(-.55,5.55)
fig.tight_layout(pad=.5,w_pad=1.8)
fig.savefig(root/'figures/public_proxy_curve.pdf',bbox_inches='tight')
plt.close(fig)
