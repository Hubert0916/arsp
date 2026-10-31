"""Generate the Random-acquisition report from the verified formative scores."""
import json
from prepare_proxy import ROOT


def main():
    d=ROOT/'results/formative'; r=ROOT/'report'
    s=json.loads((d/'summary.json').read_text())
    m=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    if s['protocol']!=m['config'] or s['n_draws']!=len(m['draws']):
        raise ValueError('Protocol mismatch')
    if s['protocol']['policies']!=['random']:
        raise ValueError('The current report requires Random acquisition only')
    methods=s['methods']['random']; pred=s['paired']['predictor_random']
    asrp=methods['full']['alc']['mean']; baseline=methods['empirical_mean']['alc']['mean']
    def save(name,text):
        if any(ord(c)<32 and c!='\n' for c in text):
            raise ValueError('Control character in report input: '+name)
        (r/name).write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
    macros={'ProxySources':s['n_distinct_sources'],'ProxyGroups':s['n_distinct_groups'],
        'ProxyDraws':s['n_draws'],'EligibleGroups':f"{s['n_eligible_groups']:,}",
        'EligibleSources':s['n_eligible_sources'],'ProxyGroupOccurrences':s['group_occurrences'],
        'ProxyTargets':f"{s['n_eval_item_occurrences_per_policy']:,}",
        'ProxyResponses':f"{s['n_eval_response_occurrences_per_policy']:,}",
        'ProxyMinGroups':s['group_count_range'][0],'ProxyMaxGroups':s['group_count_range'][1],
        'ProxyMinPairs':s['source_pair_range'][0],'ProxyMaxPairs':s['source_pair_range'][1],
        'ASRPRandomALC':f'{asrp:.4f}','EMRandomALC':f'{baseline:.4f}',
        'PredictorALCGain':f'{baseline-asrp:.4f}'}
    save('results_metadata.tex','\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items()))
    table=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'Predictor (Random acquisition) & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\',r'\midrule']
    md=['| Predictor (Random acquisition) | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |','|---|---:|---:|---:|---:|---:|---:|---:|']
    combinations={}
    for variant,name in [('empirical_mean','Empirical Mean'),('full','ASRP')]:
        x=methods[variant]; curve=[b['mean'] for b in x['brier']]; alc=x['alc']['mean']
        table.append(name+' & '+' & '.join(f'{v:.4f}' for v in curve+[alc])+r'\\')
        md.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in curve+[alc])+' |')
        combinations[('asrp' if variant=='full' else variant)+'_random']={'brier':curve,'brier_alc':alc}
    save('results_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    save('results_proxy.json',json.dumps({'budgets':s['protocol']['budgets'],'combinations':combinations},indent=2))
    save('formative_readme_table.md','\n'.join(md))
    p=ROOT/'README.md';text=p.read_text();start=text.index('<!-- RESULTS_START -->')+len('<!-- RESULTS_START -->');end=text.index('<!-- RESULTS_END -->')
    p.write_text(text[:start]+'\n\n'+'\n'.join(md)+'\n\n'+text[end:])
    table=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Predictor (Random acquisition) & Mean ALC & SD & Minimum & Maximum\\',r'\midrule']
    for variant,name in [('empirical_mean','Empirical Mean'),('full','ASRP')]:
        x=methods[variant]['alc'];table.append(name+' & '+' & '.join(f'{x[k]:.4f}' for k in ['mean','sample_sd','min','max'])+r'\\')
    save('results_seed_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    table=[r'\begin{tabular}{lrrrrrrrr}',r'\toprule',
        r'Variant & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC & $\Delta$ALC\\',r'\midrule']
    for variant,name in [('full','ASRP'),('no_rasch','w/o Rasch'),
                         ('no_kernel','w/o kernel'),('no_topic','w/o topic')]:
        x=methods[variant]; alc=x['alc']['mean']
        delta='---' if variant=='full' else f'{alc-asrp:+.6f}'
        table.append(name+' & '+' & '.join(f"{b['mean']:.4f}" for b in x['brier'])
                     +f' & {alc:.6f} & {delta}'+r'\\')
    save('results_ablation_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    rasch=s['paired']['remove_no_rasch']; kernel=s['paired']['remove_no_kernel']; topic=s['paired']['remove_no_topic']
    save('results_ablation_analysis.tex',
        r'\textbf{All removals increase mean ALC, with different budget-specific effects.} '
        +f"Topic calibration contributes most at budget 0 (Brier difference on removal: ${topic['brier_difference'][0]:+.4f}$), "
        +f"and the kernel at budget 31 (${kernel['brier_difference'][5]:+.4f}$). "
        +f"Rasch has a small mean ALC benefit (${rasch['mean']:.6f}$) and no effect at budgets 15 and 31, where its branch has zero weight. "
        +r'These benefits are not uniform across budgets or draws.')
    save('results_overview.tex',f"Across {s['n_draws']} public-data draws, \\textbf{{ASRP reaches mean Brier ALC ${asrp:.4f}$}}, compared with ${baseline:.4f}$ for Empirical Mean using identical randomly acquired labels (Table~\\ref{{tab:proxy}}). Each draw contains complete groups under the 1,000-pair cap; Table~\\ref{{tab:seeds}} reports score variation across sampled group compositions.")
    save('results_analysis.tex',r'\paragraph{Matched-evidence comparison.} '
        +f"With randomly acquired evidence held fixed, ASRP changes ALC by ${pred['mean']:+.4f}$ relative to Empirical Mean and scores lower in {pred['n_negative']} of {s['n_draws']} draws. "
        +r'The comparison measures the complete predictor under the same label-selection rule. '
        +f"The largest mean Brier reduction is at budget 1 (${ -pred['brier_difference'][1]:.4f}$), where Empirical Mean predicts only zero or one from a single label. "
        +r'ASRP instead combines its prior with the observed response. The gap narrows as more labels stabilize the empirical estimate.')
    save('introduction_results.tex',r'\textbf{Matched randomly acquired labels isolate the predictor comparison.} '
        +f"Across {s['n_draws']} complete-group public draws, ASRP achieves mean Brier ALC ${asrp:.4f}$ versus ${baseline:.4f}$ for Empirical Mean. "
        +r'All six mean budget scores favor ASRP. Component removals identify the largest topic-calibration effect at budget zero and kernel effect at budget 31, while the mean Rasch effect is small. These findings concern predictor changes with label selection held fixed; public fitting and evaluation overlap, so cold-start transfer to unseen benchmarks remains an open question.')
    names=sorted({g['source'] for g in m['groups']})
    save('source_inventory.tex',', '.join(r'\texttt{'+name.replace('_',r'\_')+'}' for name in names)+'.')
    save('conclusion_results.tex',r'ASRP combines structured initialization with support-conditioned residual adaptation as randomly acquired labels accumulate. ASRP outperforms Empirical Mean on average with matched acquired evidence in the public simulation. The released protocol, frozen artifacts, predictions, and acquisition traces make the predictor comparison reproducible.')
    print('Generated Random-acquisition report inputs.')


if __name__=='__main__':main()
