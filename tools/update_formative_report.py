"""Generate report values from the complete, fixed formative simulation."""
import json
from pathlib import Path
import statistics

from prepare_proxy import ROOT


def main():
    d=ROOT/'results/formative';r=ROOT/'report'
    s=json.loads((d/'summary.json').read_text());m=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    if s['protocol']!=m['config'] or s['n_draws']!=len(m['draws']):raise ValueError('Protocol mismatch')
    methods=s['methods'];pairs=s['paired']
    def save(name,text):(r/name).write_text(text+'\n')
    def value(policy,variant):return methods[policy][variant]['alc']['mean']
    aa=value('adaptive','full');ar=value('random','full');ea=value('adaptive','empirical_mean')
    macros={'ProxySources':s['n_distinct_sources'],'ProxyGroups':s['n_distinct_groups'],
        'ProxyDraws':s['n_draws'],'EligibleGroups':f"{s['n_eligible_groups']:,}",
        'EligibleSources':s['n_eligible_sources'],'ProxyGroupOccurrences':s['group_occurrences'],
        'ProxyTargets':f"{s['n_eval_item_occurrences_per_policy']:,}",
        'ProxyResponses':f"{s['n_eval_response_occurrences_per_policy']:,}",
        'ProxyMinGroups':s['group_count_range'][0],'ProxyMaxGroups':s['group_count_range'][1],
        'ProxyMinPairs':s['source_pair_range'][0],'ProxyMaxPairs':s['source_pair_range'][1],
        'ASRPAdaptiveALC':f'{aa:.4f}','ASRPRandomALC':f'{ar:.4f}','EMAdaptiveALC':f'{ea:.4f}',
        'PredictorALCGain':f'{ea-aa:.4f}','AcquisitionALCGain':f'{ar-aa:.4f}'}
    save('results_metadata.tex','\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items()))
    combinations={};table=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'Predictor + acquisition & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\',r'\midrule']
    md=['| Predictor + acquisition | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant,name in [('empirical_mean','Empirical Mean'),('full','ASRP')]:
        for policy in ['random','adaptive']:
            x=methods[policy][variant];curve=[b['mean'] for b in x['brier']];alc=x['alc']['mean']
            label=f'{name} + {policy.capitalize()}'
            table.append(label+' & '+' & '.join(f'{v:.4f}' for v in curve+[alc])+r'\\')
            md.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in curve+[alc])+' |')
            combinations[('asrp' if variant=='full' else variant)+'_'+policy]={'brier':curve,'brier_alc':alc}
    save('results_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    save('results_proxy.json',json.dumps({'budgets':s['protocol']['budgets'],'combinations':combinations},indent=2))
    save('formative_readme_table.md','\n'.join(md))
    readme=ROOT/'README.md';current=readme.read_text()
    start=current.index('<!-- RESULTS_START -->')+len('<!-- RESULTS_START -->')
    end=current.index('<!-- RESULTS_END -->')
    readme.write_text(current[:start]+'\n\n'+'\n'.join(md)+'\n\n'+current[end:])
    table=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Predictor + acquisition & Mean ALC & SD & Minimum & Maximum\\',r'\midrule']
    for variant,name in [('empirical_mean','Empirical Mean'),('full','ASRP')]:
        for policy in ['random','adaptive']:
            x=methods[policy][variant]['alc']
            table.append(f'{name} + {policy.capitalize()} & '+' & '.join(f'{x[k]:.4f}' for k in ['mean','sample_sd','min','max'])+r'\\')
    save('results_seed_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    save('results_overview.tex',f"Across {s['n_draws']} fixed public-data draws, \\textbf{{ASRP + Adaptive reaches mean Brier ALC ${aa:.4f}$}}, compared with ${ea:.4f}$ for Empirical Mean using the same labels (Table~\\ref{{tab:proxy}}). Each draw contains complete groups under the 1,000-pair cap; Table~\\ref{{tab:seeds}} reports variation across these changing group compositions.")
    pred=pairs['predictor_adaptive'];acq=pairs['acquisition_asrp']
    text=(r'\paragraph{Prediction and acquisition.} '
        +f"With Adaptive evidence held fixed, ASRP changes ALC by ${pred['mean']:+.4f}$ relative to Empirical Mean and scores lower in {pred['n_negative']} of {s['n_draws']} draws. "
        +f"Changing ASRP from Random to Adaptive acquisition changes mean ALC by ${acq['mean']:+.4f}$; Adaptive scores lower in {acq['n_negative']} draws, higher in {acq['n_positive']}, and ties in {acq['n_zero']}. "
        +r'These paired comparisons distinguish the predictor contribution from the acquisition-policy contribution; draw-level SD describes the sampled public group compositions.')
    save('results_analysis.tex',text)
    acquisition_word='reduces' if ar>aa else 'increases'
    save('introduction_results.tex',r'\textbf{The evaluation separates prediction quality from label selection.} '
        +f"Across {s['n_draws']} complete-group public draws, ASRP achieves mean Brier ALC ${aa:.4f}$ versus ${ea:.4f}$ for Empirical Mean under identical Adaptive evidence. "
        +f"Switching ASRP from Random to Adaptive acquisition {acquisition_word} mean ALC by ${abs(ar-aa):.4f}$. "
        +r'Component ablations find that Rasch has mixed group-dependent effects, while kernel adaptation and topic calibration yield small mean improvements under Adaptive evidence. '
        +r'Ten official evaluations yield mean ALC $\OfficialALC{}$, but expose a \textbf{cold-start transfer weakness}: zero-label Brier is $\OfficialZeroBrier{}$, above the $0.2500$ of a constant $0.5$ prediction.')
    table=[r'\begin{tabular}{lrrrrrrrr}',r'\toprule',r'Adaptive predictor & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC & $\Delta$ALC\\',r'\midrule']
    for variant,label in [('full','Full ASRP'),('no_rasch','No Rasch'),('no_kernel','No Kernel Probe'),('no_topic','No topic calibration')]:
        x=methods['adaptive'][variant];vals=[b['mean'] for b in x['brier']]+[x['alc']['mean']]
        table.append(label+' & '+' & '.join(f'{v:.4f}' for v in vals)+f" & {x['alc']['mean']-aa:+.4f}"+r'\\')
    save('ablation_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    texts=[]
    for v,label in [('no_rasch','Rasch'),('no_kernel','Kernel Probe'),('no_topic','topic calibration')]:
        p=pairs['remove_'+v]
        texts.append(f"Removing {label} changes mean ALC by ${p['mean']:+.5f}$, worsening {p['n_positive']}/{s['n_draws']} draws. ")
    rasch=pairs['remove_no_rasch'];delta=rasch['brier_difference']
    texts.append(f"For Rasch, the removal effects at budgets zero and one are ${delta[0]:+.5f}$ and ${delta[1]:+.5f}$, respectively. ")
    texts.append(r'Rasch removal leaves predictions unchanged at budgets 15 and 31, where its mixture coefficient is zero. These matched-evidence effects concern the frozen system without refitting.')
    save('ablation_analysis.tex',r'\paragraph{Controlled component effects.} '+''.join(texts))
    names=sorted({g['source'] for g in m['groups']})
    save('source_inventory.tex',', '.join(r'\texttt{'+name.replace('_',r'\_')+'}' for name in names)+'.')
    group_rows=json.loads((d/'by_group.json').read_text())
    unique={}
    for row in group_rows:
        if row['policy']=='adaptive':unique.setdefault(row['group_id'],[]).append(row)
    effects=[]
    for gid,rows in unique.items():
        effects.append({'group_id':gid,'source':rows[0]['source'],'subject_id':rows[0]['subject_id'],
            'n_draws':len(rows),'with_rasch':statistics.mean(x['methods']['full']['alc'] for x in rows),
            'without_rasch':statistics.mean(x['methods']['no_rasch']['alc'] for x in rows)})
    for e in effects:e['delta']=e['without_rasch']-e['with_rasch']
    effects.sort(key=lambda e:e['delta'])
    (d/'rasch_by_group.json').write_text(json.dumps(effects,indent=2)+'\n')
    win=sum(e['delta']>0 for e in effects);loss=sum(e['delta']<0 for e in effects)
    save('rasch_retention_analysis.tex',r'\paragraph{Group heterogeneity.} '+f"Averaging repeated appearances per group, retaining Rasch improves ALC in {win}/{len(effects)} distinct groups and worsens it in {loss}. "+f"ASRP + Adaptive draw scores range from ${methods['adaptive']['full']['alc']['min']:.4f}$ to ${methods['adaptive']['full']['alc']['max']:.4f}$. "+r'Only one to five complete groups fit each public draw, making scores sensitive to group composition. This variation does not quantify private-test uncertainty.')
    save('conclusion_results.tex',r'ASRP combines structured initialization with support-conditioned residual adaptation as labels accumulate. '
        +('ASRP outperforms Empirical Mean on average with matched acquired evidence. ' if aa<ea else 'Matched-evidence comparisons expose the limits of the current predictor. ')
        +r'Component and acquisition effects vary across sampled groups, while accurate cold-start transfer remains a challenge. The released protocol, frozen artifacts, predictions, and acquisition traces make these comparisons reproducible.')
    print('Generated formative report tables, macros, analyses, and group diagnostics.')


if __name__=='__main__':main()
