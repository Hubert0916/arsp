"""Quantify paired Rasch effects and draw-composition uncertainty.

Intervals are percentile bootstrap intervals over the 30 observed public draws,
conditional on their fixed public snapshot, not private-test uncertainty.
"""
import json
from pathlib import Path

import numpy as np

from prepare_proxy import ROOT, sha, write_json


def main():
    directory=ROOT/'results/formative'
    draws=json.loads((directory/'draws.json').read_text())
    groups=json.loads((directory/'by_group.json').read_text())
    manifest=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    benchmarks={g['group_id']:g['benchmark_alias'] for g in manifest['groups']}
    rows=[r for r in draws if r['policy']=='adaptive']
    delta=np.array([r['methods']['no_rasch']['alc']-r['methods']['full']['alc'] for r in rows])
    curves=np.array([[a-b for a,b in zip(r['methods']['no_rasch']['brier'],r['methods']['full']['brier'])] for r in rows])
    rng=np.random.default_rng(20261031)
    ix=rng.integers(0,len(rows),size=(100000,len(rows)))
    interval=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()
    budget_intervals=[np.quantile(curves[:,j][ix].mean(1),[.025,.975]).tolist() for j in range(6)]
    weights={r['draw_id']:1/(len(rows)*r['n_groups']) for r in rows}
    contributions=[]
    for r in groups:
        if r['policy']!='adaptive':continue
        difference=r['methods']['no_rasch']['alc']-r['methods']['full']['alc']
        contributions.append({
            'draw_id':r['draw_id'],'group_id':r['group_id'],'source':r['source'],
            'with_rasch_alc':r['methods']['full']['alc'],
            'without_rasch_alc':r['methods']['no_rasch']['alc'],
            'difference':difference,'aggregation_weight':weights[r['draw_id']],
            'weighted_contribution':weights[r['draw_id']]*difference,
            'brier_difference':[a-b for a,b in zip(r['methods']['no_rasch']['brier'],r['methods']['full']['brier'])]})
    assert abs(sum(x['aggregation_weight'] for x in contributions)-1)<1e-12
    positive=sum(x['weighted_contribution'] for x in contributions if x['difference']>0)
    negative=sum(x['weighted_contribution'] for x in contributions if x['difference']<0)
    assert abs(positive+negative-delta.mean())<1e-12
    best=max(rows,key=lambda r:r['methods']['no_rasch']['alc'])
    result={
        'definition':'No Rasch minus full ASRP; positive means retaining Rasch helps.',
        'n_draws':len(rows),'mean_delta':float(delta.mean()),'sample_sd_delta':float(delta.std(ddof=1)),
        'bootstrap_seed':20261031,'bootstrap_replicates':100000,'percentile_95_interval':interval,
        'interval_scope':'Resample whole paired public draws; conditional on the fixed public snapshot and frozen artifacts. Not private-test uncertainty or proof of equivalence.',
        'n_draws_retention_helps':int(sum(delta>0)),'n_draws_retention_hurts':int(sum(delta<0)),
        'n_draws_with_repeated_benchmarks':sum(len({benchmarks[g] for g in r['group_ids']})<r['n_groups'] for r in rows),
        'budget_differences':curves.mean(0).tolist(),'pointwise_budget_95_intervals':budget_intervals,
        'positive_weighted_group_contributions':positive,'negative_weighted_group_contributions':negative,
        'worst_no_rasch_public_draw':{'draw_id':best['draw_id'],'n_groups':best['n_groups'],
            'with_rasch_alc':best['methods']['full']['alc'],'without_rasch_alc':best['methods']['no_rasch']['alc']},
        'input_sha256':{n:sha(directory/n) for n in ['draws.json','by_group.json','summary.json']},
        'formative_manifest_sha256':sha(ROOT/'experiments/formative_manifest.json'),
        'script_sha256':sha(Path(__file__))}
    output=ROOT/'results/rasch_audit'
    write_json(output/'local_effects.json',result)
    write_json(output/'group_contributions.json',contributions)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
