"""Explain paired Rasch effects by budget and source, with fixed inference artifacts."""
from collections import defaultdict
import json
import statistics
from prepare_proxy import ROOT, sha


def main():
    initial = json.loads((ROOT/'results/ablation/summary.json').read_text())['variants']
    extra = json.loads((ROOT/'results/rasch_confirmation/summary.json').read_text())['variants']
    with_rasch = initial['full']['per_seed'] + extra['legacy_rasch']['per_seed']
    without = initial['no_rasch']['per_seed'] + extra['without_rasch']['per_seed']
    if any([r['seed'] for r in rows] != list(range(42,48)) for rows in (with_rasch,without)):
        raise ValueError('Unexpected paired seed coverage')
    metrics={}
    for name,rows in [('with_rasch',with_rasch),('without_rasch',without)]:
        metrics[name]={'brier':[statistics.mean(r['brier'][j] for r in rows) for j in range(6)],
                       'alc':statistics.mean(r['alc'] for r in rows)}
    full,removed=metrics['with_rasch'],metrics['without_rasch']
    first=json.loads((ROOT/'results/ablation/by_source.json').read_text())
    other=json.loads((ROOT/'results/rasch_confirmation/by_source.json').read_text())
    lookup={(r['source'],r['seed'],r['variant']):r for r in first}
    pairs=[(s,seed,row,lookup[(s,seed,'no_rasch')]) for (s,seed,v),row in lookup.items() if v=='full']
    pairs += [(r['source'],r['seed'],r['legacy_rasch'],r['without_rasch']) for r in other]
    grouped=defaultdict(dict)
    for source,seed,a,b in pairs:
        if seed in grouped[source]:raise ValueError('Duplicate source/seed')
        grouped[source][seed]=[a['brier'][j]-b['brier'][j] for j in range(6)]
    if len(grouped)!=92 or any(set(v)!=set(range(42,48)) for v in grouped.values()):
        raise ValueError('Incomplete source/seed comparison')
    source_deltas=[{'source':s,'delta_brier':[statistics.mean(v[j] for v in by_seed.values()) for j in range(6)]}
                   for s,by_seed in sorted(grouped.items())]
    weights=[.1,.2,.2,.2,.2,.1];budgets=[0,1,3,7,15,31]
    config=json.loads((ROOT/'artifacts/asrp/runtime_config.json').read_text())
    effects=[]
    for j,n in enumerate(budgets):
        d=full['brier'][j]-removed['brier'][j]
        values=[r['delta_brier'][j] for r in source_deltas]
        if abs(statistics.mean(values)-d)>1e-12:raise ValueError('Budget/source aggregates disagree')
        effective=(1-config['probe_weights'][j])*config['lambdas'][j]
        effects.append({'budget':n,'with_rasch':full['brier'][j],'without_rasch':removed['brier'][j],
                        'delta_with_minus_without':d,'alc_weight':weights[j],
                        'alc_contribution':weights[j]*d,'effective_rasch_weight':effective,
                        'source_wins':sum(v < -1e-12 for v in values),'source_losses':sum(v > 1e-12 for v in values),
                        'source_ties':sum(abs(v)<=1e-12 for v in values),
                        'mean_absolute_source_effect':statistics.mean(abs(v) for v in values)})
    if abs(sum(r['alc_contribution'] for r in effects)-(full['alc']-removed['alc']))>1e-12:
        raise ValueError('ALC decomposition mismatch')
    if full['brier'][-2:]!=removed['brier'][-2:]:raise ValueError('Kernel-only checkpoints changed')
    analysis={'definition':'Differences are with Rasch minus without Rasch; negative is favorable to Rasch.',
              'seeds':list(range(42,48)),'n_sources':92,'budget_effects':effects,
              'delta_alc':full['alc']-removed['alc'],
              'scope':'Paired component removal on the same public sources and frozen calibration; not source-held-out validation.'}
    for name,value in [('budget_analysis.json',analysis),('budget_by_source.json',source_deltas)]:
        (ROOT/'results/rasch_confirmation'/name).write_text(json.dumps(value,indent=2)+'\n')
    zero_wins=sum(a['brier'][0]<b['brier'][0] for a,b in zip(with_rasch,without))
    decision={'action':'retain_rasch',
        'reason':'Keep the configured Rasch contribution for observed zero-label initialization and budget-dependent complementary predictions. Local low-budget effects are mixed; retention is not a claim of necessity or a proven overall ALC advantage.',
        'runtime_reference_commit':'f9afbd258cc1f47ef158487261c79db14a48eafa',
        'inference_manifest_sha256':sha(ROOT/'artifacts/manifest.sha256.json'),
        'seeds':list(range(42,48)),'metrics':metrics,'with_rasch_zero_label_wins':zero_wins,
        'with_rasch_alc_wins':sum(a['alc']<b['alc'] for a,b in zip(with_rasch,without)),
        'scope':analysis['scope'],'budget_analysis':'results/rasch_confirmation/budget_analysis.json',
        'prior_ablation':'results/rasch_prior_ablation/summary.json'}
    (ROOT/'results/rasch_confirmation/decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    table=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',
           r'Paired variant & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\',r'\midrule']
    for label,value in [('With Rasch',full),('Without Rasch',removed)]:
        table.append(label+' & '+' & '.join(f'{v:.5f}' for v in value['brier']+[value['alc']])+r'\\')
    table.append(r'$\Delta$ (with $-$ without) & '+' & '.join(f'${r["delta_with_minus_without"]:+.5f}$' for r in effects)
                 +f" & ${analysis['delta_alc']:+.5f}$"+r'\\')
    table += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'report/rasch_retention_table.tex').write_text('\n'.join(table)+'\n')
    text=(r'\paragraph{Rasch has budget-dependent effects.} '
          +f"Across six paired draws, Rasch lowers zero-label Brier from ${removed['brier'][0]:.5f}$ to ${full['brier'][0]:.5f}$ "
          +f"in the aggregate, with this direction in all {zero_wins} seeds. It also slightly improves budget 7, but worsens budgets 1 and 3 (Table~\\ref{{tab:rasch_retention}}). "
          +f"\\textbf{{Aggregate ALC hides these offsetting effects}}: the budget-zero gain contributes ${effects[0]['alc_contribution']:+.6f}$ to ALC, "
          +f"while budgets one and three contribute ${effects[1]['alc_contribution']:+.6f}$ and ${effects[2]['alc_contribution']:+.6f}$, respectively. "
          +f"The net ALC change is ${analysis['delta_alc']:+.6f}$. "
          +r'At budgets 15 and 31, $(1-w_n)\eta_n=0$, so this ablation cannot assess Rasch there. '
          +f"Effects also vary by source: at budget zero, Rasch improves {effects[0]['source_wins']} of 92 source means and worsens {effects[0]['source_losses']}. "
          +r'The evidence supports an initialization contribution in the frozen configuration, rather than a uniform advantage throughout the few-label regime.')
    (ROOT/'report/rasch_retention_analysis.tex').write_text(text+'\n')
    print('Generated budget-wise ALC decomposition and complete source effect counts.')


if __name__=='__main__':main()
