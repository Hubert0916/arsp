"""Plot the four paired predictor/acquisition curves from scored results."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).parent
res=json.loads((root/'results_proxy.json').read_text())
x=np.log2(np.asarray(res['budgets'],dtype=float)+1)
plt.rcParams.update({'font.size':8,'axes.labelsize':8.5,'xtick.labelsize':7.5,
                     'ytick.labelsize':7.5,'legend.fontsize':6.5,'axes.linewidth':.8})
fig,ax=plt.subplots(figsize=(3.25,2.05))
values=[]
for method,name,color in [('asrp','ASRP','#1f5a9d'),('empirical_mean','Empirical Mean','#b35c1e')]:
 for policy,style,marker in [('random','--','s'),('adaptive','-','o')]:
  y=res['combinations'][method+'_'+policy]['brier'];values.extend(y)
  ax.plot(x,y,linestyle=style,marker=marker,linewidth=1.25,markersize=3,
          color=color,label=f'{name}, {policy}')
ax.set_xticks(x,[str(n) for n in res['budgets']])
ax.set_xlabel('Label budget, $n$',labelpad=2);ax.set_ylabel('Brier score',labelpad=2)
ax.set_ylim(min(values)-.015,max(values)+.02)
ax.grid(axis='y',color='#d9d9d9',linewidth=.55)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(width=.7,length=3,pad=2)
ax.legend(frameon=False,loc='upper right',handlelength=2,borderaxespad=.2,labelspacing=.18)
fig.tight_layout(pad=.35)
fig.savefig(root/'figures/public_proxy_curve.pdf',bbox_inches='tight')
