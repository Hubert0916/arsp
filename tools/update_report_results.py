"""Generate all reported numbers from the scored paired acquisition experiment."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_proxy import ROOT

ORDER=['empirical_mean_random','empirical_mean_adaptive','asrp_random','asrp_adaptive']
LABELS={'empirical_mean_random':'Empirical Mean + Random','empirical_mean_adaptive':'Empirical Mean + Adaptive',
        'asrp_random':'ASRP + Random','asrp_adaptive':'ASRP + Adaptive'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=ROOT/'results/proxy')
    args=parser.parse_args()
    result=json.loads((args.directory/'summary.json').read_text())
    manifest=json.loads((ROOT/'experiments/proxy_manifest.json').read_text())
    if result['n_groups']!=len(manifest['groups']) or result['seeds']!=manifest['config']['seeds']:
        raise ValueError('Only a complete published run can update the report')
    shutil.copyfile(args.directory/'summary.json',ROOT/'report/results_proxy.json')
    root=ROOT/'report';c=result['combinations']
    def save(name,text): (root/name).write_text(text+'\n')
    macros={'ProxySources':result['n_sources'],'ProxyItems':f"{result['n_items']:,}",
            'ProxyTargets':f"{result['n_rows_per_seed']:,}",'ProxyExcluded':result['n_excluded_sources'],
            'ProxyMinItems':result['group_items_range'][0],'ProxyMaxItems':result['group_items_range'][1],
            'ProxyMinPool':result['acquisition_pool_range'][0],'ProxyMaxPool':result['acquisition_pool_range'][1]}
    save('results_metadata.tex','\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items()))
    header=r'\begin{tabular}{lrrrrrrr}'+'\n'+r'\toprule'+'\n'+r'Predictor + acquisition & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\'+'\n'+r'\midrule'+'\n'
    rows=''.join(LABELS[k]+' & '+' & '.join(f'{v:.4f}' for v in c[k]['brier']+[c[k]['alc']])+r'\\'+'\n' for k in ORDER)
    save('results_table.tex',header+rows+r'\bottomrule'+'\n'+r'\end{tabular}')
    seed_table=r'\begin{tabular}{lrrrr}'+'\n'+r'\toprule'+'\n'+r'Predictor + acquisition & Seed 42 & Seed 43 & Seed 44 & SD\\'+'\n'+r'\midrule'+'\n'
    for k in ORDER:
        seed_table+=LABELS[k]+' & '+' & '.join(f"{c[k]['per_seed_alc'][str(s)]:.4f}" for s in result['seeds'])+f" & {c[k]['alc_seed_sd']:.4f}"+r'\\'+'\n'
    save('results_seed_table.tex',seed_table+r'\bottomrule'+'\n'+r'\end{tabular}')
    a,b=c['asrp_adaptive'],c['empirical_mean_random']
    wins=sum(x<y for x,y in zip(a['brier'],b['brier']));gain=b['alc']-a['alc']
    overview=(r'Table~\ref{tab:proxy} reports the \textbf{four predictor--acquisition combinations}, averaged equally over groups and then over seeds 42, 43, and 44. '
              +f"ASRP + Adaptive has a mean Brier ALC of ${a['alc']:.4f}$, compared with ${b['alc']:.4f}$ for Empirical Mean + Random "
              +f"(an absolute {'reduction' if gain>=0 else 'increase'} of ${abs(gain):.4f}$). "
              +f'Its aggregate Brier score is lower at {wins} of the six budgets. '
              +r'The two predictors share identical labels within each acquisition policy; the Random and Adaptive policies use identical candidate orders and evaluation pools.')
    save('results_overview.tex',overview)
    analysis=r'\paragraph{Predictor effect with acquisition held fixed.}'+'\n'
    for policy in ['random','adaptive']:
        x,y=c['asrp_'+policy],c['empirical_mean_'+policy];d=y['alc']-x['alc']
        analysis+=f"Under {policy.capitalize()} acquisition, ASRP's mean ALC is ${x['alc']:.4f}$ versus ${y['alc']:.4f}$ for Empirical Mean, "
        analysis+=f"a {'reduction' if d>=0 else 'increase'} of ${abs(d):.4f}$. "
    analysis+=f"Empirical Mean + Random rises from ${b['brier'][0]:.4f}$ at zero labels to ${b['brier'][1]:.4f}$ at one label: a single outcome makes its prediction either zero or one. ASRP smooths early evidence using public priors. "
    analysis+=r'At budget zero the acquisition policies have no effect. The Rasch hybrid followed by topic correction supplies the initial prediction; the kernel branch receives increasing weight as labels become available.'+'\n\n'
    analysis+=r'\paragraph{Acquisition effect with the predictor held fixed.}'+'\n'
    for method,label in [('asrp','ASRP'),('empirical_mean','Empirical Mean')]:
        e=result['acquisition_effect'][method];d=e['adaptive_minus_random_alc']
        seed_phrase="all three seeds" if e["seeds_improved"]==3 else f"{e['seeds_improved']} of the three seeds"
        analysis+=f"For {label}, Adaptive {'lowers' if d<0 else 'raises'} mean ALC by ${abs(d):.4f}$ relative to Random, and has lower ALC in {seed_phrase}. "
    smallest_seed=min(result['acquisition_effect']['asrp']['per_seed'],key=lambda seed:abs(result['acquisition_effect']['asrp']['per_seed'][seed]))
    smallest_diff=result['acquisition_effect']['asrp']['per_seed'][smallest_seed]
    analysis+=f"The closest ASRP pair differs by only ${smallest_diff:+.5f}$ (Adaptive minus Random, seed {smallest_seed}). "
    analysis+=r'Table~\ref{tab:seeds} shows the individual seed results and their sample standard deviation; three seeds summarize sensitivity to these draws rather than establishing universal superiority.'+'\n\n'
    analysis+=r'\paragraph{Selection capacity.}'+'\n'
    analysis+=f"Across the full experiment, {result['adaptive_trajectories_with_discretion']} of {result['n_groups']*len(result['seeds'])} Adaptive trajectories execute at least one discretionary decision. "
    max_budget=max(result['budgets'])
    forced_groups=[g for g in manifest['groups'] if (g['n_items']+1)//2-min(16,max(0,(g['n_items']+1)//2-max_budget))<=max_budget]
    if len(forced_groups)==1:
        size=forced_groups[0]['n_items'];warmup=(size+1)//2-max_budget
        analysis+=f"The {len(result['seeds'])} trajectories without discretionary choices belong to one {size}-item group: {warmup} warm-up skips leave exactly {max_budget} candidates, all mandatory. "
    analysis+=r'Topic calibration is active; the source-name guard remains inactive because the benchmark aliases are anonymous.'
    save('results_analysis.tex',analysis)
    d=result['acquisition_effect']['asrp']['adaptive_minus_random_alc']
    ending=(f"The three-seed public proxy gives ASRP + Adaptive a mean Brier ALC of ${a['alc']:.4f}$. "
            +f"Compared with Random acquisition using the same predictor, Adaptive {'reduces' if d<0 else 'increases'} mean ALC by ${abs(d):.4f}$. "
            +r'The paired design separates changes in prediction quality from changes in which labels are acquired.')
    save('results_conclusion.tex',ending)
    save('source_inventory.tex',r'\begin{verbatim}'+'\n'+'\n'.join(g['source'] for g in manifest['groups'])+'\n'+r'\end{verbatim}')
    markdown=['| Predictor + acquisition | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ORDER: markdown.append('| '+LABELS[k]+' | '+' | '.join(f'{v:.4f}' for v in c[k]['brier']+[c[k]['alc']])+' |')
    readme=ROOT/'README.md';prefix,remainder=readme.read_text().split('<!-- RESULTS_START -->',1);_,suffix=remainder.split('<!-- RESULTS_END -->',1)
    readme.write_text(prefix+'<!-- RESULTS_START -->\n\n'+'\n'.join(markdown)+'\n\n<!-- RESULTS_END -->'+suffix)
    print('Updated report and README from the complete paired results.')


if __name__=='__main__':main()
