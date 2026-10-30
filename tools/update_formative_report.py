"""Generate report values from the complete, fixed formative simulation."""
import json
from pathlib import Path
import statistics

from prepare_proxy import ROOT, sha
from update_official_report import official_results_enabled


def main():
    d=ROOT/'results/formative';r=ROOT/'report'
    s=json.loads((d/'summary.json').read_text());m=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    if s['protocol']!=m['config'] or s['n_draws']!=len(m['draws']):raise ValueError('Protocol mismatch')
    methods=s['methods'];pairs=s['paired']
    audit_dir=ROOT/'results/rasch_audit'
    audit=json.loads((audit_dir/'local_effects.json').read_text())
    version=json.loads((audit_dir/'version_check.json').read_text())
    context=json.loads((audit_dir/'context_check.json').read_text())
    for name,digest in audit['input_sha256'].items():
        if sha(d/name)!=digest:raise ValueError('Rasch audit inputs changed: '+name)
    def save(name,text):
        if any(ord(c)<32 and c!='\n' for c in text):
            raise ValueError('Unexpected control character in generated report file: '+name)
        (r/name).write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
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
        +(r'Ten official evaluations yield mean ALC $\OfficialALC{}$, but expose a \textbf{cold-start transfer weakness}: zero-label Brier is $\OfficialZeroBrier{}$, above the $0.2500$ of a constant $0.5$ prediction.'
          if official_results_enabled() else ''))
    table=[r'\begin{tabular}{lrrrrrrrr}',r'\toprule',r'Adaptive predictor & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC & $\Delta$ALC\\',r'\midrule']
    for variant,label in [('full','Full ASRP'),('no_rasch','No Rasch'),('no_kernel','No Kernel Probe'),('no_topic','No topic calibration')]:
        x=methods['adaptive'][variant];vals=[b['mean'] for b in x['brier']]+[x['alc']['mean']]
        table.append(label+' & '+' & '.join(f'{v:.4f}' for v in vals)+f" & {x['alc']['mean']-aa:+.4f}"+r'\\')
    save('ablation_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    texts=[f"Removing Kernel Probe and topic calibration increases mean ALC by ${pairs['remove_no_kernel']['mean']:.5f}$ and ${pairs['remove_no_topic']['mean']:.5f}$, respectively. "]
    rasch=pairs['remove_no_rasch'];delta=rasch['brier_difference']
    lo,hi=audit['percentile_95_interval']
    texts.append(f"Rasch removal changes mean ALC by ${rasch['mean']:+.5f}$; a paired-draw 95\\% bootstrap interval is $[{lo:.5f}, {hi:.5f}]$. ")
    texts.append(r'This interval spans zero and establishes neither an average advantage nor equivalence. Zero Rasch weight makes predictions identical at budgets 15 and 31. See Appendix~\ref{app:rasch_audit} for resampling and diagnostic checks.')
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
    save('rasch_retention_analysis.tex',r'\paragraph{Group heterogeneity.} '+f"Averaging repeated appearances per group, retaining Rasch improves ALC in {win}/{len(effects)} distinct groups and worsens it in {loss}. "+r'Only one to five complete groups fit each public draw, making aggregate scores sensitive to composition. Per-group results accompany the release.')
    save('conclusion_results.tex',r'ASRP combines structured initialization with support-conditioned residual adaptation as labels accumulate. '
        +('ASRP outperforms Empirical Mean on average with matched acquired evidence. ' if aa<ea else 'Matched-evidence comparisons expose the limits of the current predictor. ')
        +r'Component and acquisition effects vary across sampled groups, while accurate cold-start transfer remains a challenge. The released protocol, frozen artifacts, predictions, and acquisition traces make these comparisons reproducible.')
    table=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Source & $B_{15}$, focal & $B_{15}$, + companion & $B_{31}$, focal & $B_{31}$, + companion\\',r'\midrule']
    for row in context['results']:
        vals=[row['focal_only_brier'][4],row['with_companion_brier'][4],row['focal_only_brier'][5],row['with_companion_brier'][5]]
        table.append(r'\texttt{'+row['source'].replace('_',r'\_')+'} & '+' & '.join(f'{v:.5f}' for v in vals)+r'\\')
    table += [r'\bottomrule',r'\end{tabular}']
    details=(r'\paragraph{Paired uncertainty and heterogeneous effects.} '
        +f"We resample the {s['n_draws']} complete paired public draws {audit['bootstrap_replicates']:,} times (seed {audit['bootstrap_seed']}) and take the 2.5th and 97.5th percentiles of the mean ALC difference. "
        +f"Rasch retention helps in {audit['n_draws_retention_helps']} draws and hurts in {audit['n_draws_retention_hurts']}. "
        +f"Under the report's group/draw weighting, group occurrences favoring retention contribute ${audit['positive_weighted_group_contributions']:+.5f}$ and those favoring removal contribute ${audit['negative_weighted_group_contributions']:+.5f}$ to the removal-minus-full difference. "
        +r'The near-zero mean results from opposing effects, not identical predictions. The interval is conditional on this public snapshot and does not estimate private-test uncertainty. '
        +f"The worst no-Rasch public draw already has ALC ${audit['worst_no_rasch_public_draw']['without_rasch_alc']:.5f}$, illustrating why a single difficult draw should not be compared with a multi-draw mean.\n\n"
        +r'\paragraph{Archived implementation check.} '
        +fr"We reconstruct the archived no-Rasch implementation (Git \texttt{{{version['archived_commit'][:7]}}}) and compare it with setting the original Rasch weights to zero. The first and last distinct evaluation inputs of each group, at all six budgets and under both policies, give {version['n_predictions']:,} checked predictions with maximum difference ${version['max_error']:.0f}$. "
        +r'The acquisition code is unchanged and the kernel implementation is semantically identical. This checks the archived code on these inputs; it does not authenticate an unavailable historical platform ZIP. '
        +r'The report retains the submitted with-Rasch configuration and does not infer that Rasch is universally necessary.'+'\n\n'
        +r'\paragraph{Shared-benchmark context.} '
        +r'All main public draws happen to contain distinct benchmarks. In draws containing multiple subjects within a benchmark, Kernel Probe and acquisition can share evidence or seen candidates across those subjects. Comparing one subject across separate draws therefore need not hold its full context fixed, even when its own item pool is persistent. '+'\n\n'
        +r'We isolate this mechanism using the first three eligible source/benchmark names in lexical order, taking the first group-ID pair with distinct visible subjects that fits the 1,000-pair cap. We hold the focal subject\textquotesingle s evaluation items and acquired-label prefixes fixed, then append only the companion\textquotesingle s acquired prefix at each budget. Acquisition is not rerun jointly. '
        +r'Budgets zero and one, and Empirical Mean, stay unchanged; kernel-dependent predictions can change even at budgets 15 and 31, where Rasch is absent. This establishes context sensitivity, not the cause or magnitude of a private-platform gap.'+'\n\n'
        +r'\begin{table}[H]'+'\n'+r'\centering\small'+'\n'+'\n'.join(table)+'\n'+r'\vspace{6pt}'+'\n'
        +r'\caption{Controlled context diagnostic: fixed focal labels and queries, with or without another subject\textquotesingle s evidence from the same benchmark. These cases are not averaged into the main results.}'+'\n'
        +r'\label{tab:context_audit}'+'\n'+r'\end{table}'+'\n\n'
        +r'Results and reproducible checks are in \path{results/rasch_audit/}, with \path{tools/audit_rasch_effects.py}, \path{tools/check_rasch_versions.py}, and \path{tools/check_shared_context.py}.')
    save('rasch_audit.tex',details)
    print('Generated formative report tables, macros, analyses, and group diagnostics.')


if __name__=='__main__':main()
