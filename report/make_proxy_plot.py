"""Show absolute Adaptive scores and paired acquisition differences."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).parent
res=json.loads((root/'results_proxy.json').read_text())
x=np.log2(np.asarray(res['budgets'],dtype=float)+1)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,
                     'ytick.labelsize':7.5,'legend.fontsize':7,'axes.linewidth':.8,
                     'pdf.fonttype':42})
fig,(score_ax,delta_ax)=plt.subplots(ncols=2,figsize=(6.4,2.15))
methods=[('asrp','ASRP','#1f5a9d','o'),
         ('empirical_mean','Empirical Mean','#b35c1e','s')]
width=.32
differences=[]
for index,(method,name,color,marker) in enumerate(methods):
 adaptive=np.asarray(res['combinations'][method+'_adaptive']['brier'])
 random=np.asarray(res['combinations'][method+'_random']['brier'])
 score_ax.plot(x,adaptive,marker=marker,linewidth=1.4,markersize=3.5,
               color=color,label=name)
 # Report differences in thousandths, without rounding the plotted values.
 delta=1000*(adaptive-random)
 differences.append(delta)
 bars=delta_ax.bar(x+(index-.5)*width,delta,width=width,color=color,
                  edgecolor=color if index==0 else 'white',linewidth=.5,
                  hatch='' if index==0 else '///')
 delta_ax.bar_label(bars,labels=[f'{value:.1f}' if value!=0 else '' for value in delta],
                    padding=2,fontsize=6.5)
for position,values in zip(x,np.asarray(differences).T):
 if np.all(values==0):
  delta_ax.annotate('0.0',(position,0),xytext=(0,3),textcoords='offset points',
                    ha='center',va='bottom',fontsize=6.5)
score_ax.set_title('(a) Brier score: Adaptive',loc='left',fontsize=8.5,pad=7)
score_ax.set_ylabel('Brier score (lower is better)',labelpad=3)
score_ax.set_ylim(.155,.365)
score_ax.set_yticks([.16,.20,.24,.28,.32,.36])
score_ax.legend(frameon=False,loc='upper right',handlelength=1.5,
                borderaxespad=.1,labelspacing=.3)
delta_ax.set_title('(b) Adaptive $-$ Random',loc='left',fontsize=8.5,pad=7)
delta_ax.set_ylabel(r'$\Delta$Brier ($\times 10^{-3}$)',labelpad=3)
delta_ax.set_ylim(-3.5,4.4)
delta_ax.set_yticks([-3,-2,-1,0,1,2,3,4])
delta_ax.axhline(0,color='#454545',linewidth=.8,zorder=3)
delta_ax.text(.98,.97,'Positive: Adaptive worse',transform=delta_ax.transAxes,
              ha='right',va='top',fontsize=6.5,color='#444444')
delta_ax.text(.98,.03,'Negative: Adaptive better',transform=delta_ax.transAxes,
              ha='right',va='bottom',fontsize=6.5,color='#444444')
for ax in (score_ax,delta_ax):
 ax.set_xticks(x,[str(n) for n in res['budgets']])
 ax.set_xlabel('Label budget, $n$',labelpad=2)
 ax.grid(axis='y',color='#e0e0e0',linewidth=.55)
 ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(width=.7,length=3,pad=2)
 ax.set_xlim(-.55,5.55)
fig.tight_layout(pad=.5,w_pad=1.8)
fig.savefig(root/'figures/public_proxy_curve.pdf',bbox_inches='tight')
plt.close(fig)
