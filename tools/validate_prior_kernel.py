"""Evaluate exactly one locked no-Rasch candidate on the untouched public groups."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_proxy import ROOT,sha,write_json
from search_prior_kernel import CANDIDATE,ALC,curve,load_cache


def paired(values,seed):
    x=np.asarray(values);rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(x),(50000,len(x)))
    return {'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),
        'percentile_95_interval':np.quantile(x[ix].mean(1),[.025,.975]).tolist(),
        'wins':int(sum(x<0)),'losses':int(sum(x>0)),'ties':int(sum(x==0))}


def main():
    lock_path=ROOT/'experiments/asrp_next/locked_candidate.json'
    lock=json.loads(lock_path.read_text())
    assert lock['implementation_sha256']==sha(ROOT/'experiments/asrp_next/prior_kernel.py')
    assert lock['protocol_sha256']==sha(ROOT/'experiments/asrp_next/protocol.json')
    assert lock['search_result_sha256']==sha(ROOT/'results/asrp_next/development/search.json')
    config=lock['config'];protocol=json.loads((ROOT/'experiments/asrp_next/protocol.json').read_text())
    manifest_path=ROOT/'experiments/asrp_next/validation_manifest.json'
    manifest_bytes=(manifest_path.read_bytes() if manifest_path.exists()
                    else gzip.decompress(manifest_path.with_suffix('.json.gz').read_bytes()))
    manifest=json.loads(manifest_bytes)
    cohorts={d['draw_id']:d['cohort'] for d in manifest['draws']}
    assert not {g['group_id'] for g in manifest['groups']}&set(protocol['validation_excluded_group_ids'])
    cache,index=load_cache(ROOT/'results/asrp_next/validation_cache')
    assert index['manifest_sha256']==hashlib.sha256(manifest_bytes).hexdigest()
    assert len(cache)==2*len(manifest['draws'])
    assert {(r['draw_id'],r['policy']) for r,_ in cache}=={
        (d['draw_id'],policy) for d in manifest['draws'] for policy in ['random','adaptive']}
    names=['candidate','full','no_rasch','em']+CANDIDATE.ablations(config)
    rows=[]
    for r,parts in cache:
        results={}
        for name in names:
            p=(parts[name] if name in ['full','no_rasch','em'] else CANDIDATE.probabilities(parts,config,None if name=='candidate' else name))
            b=curve(parts,p);results[name]={'brier':b.tolist(),'alc':float(b@ALC)}
        rows.append({'draw_id':r['draw_id'],'cohort':cohorts[r['draw_id']],'policy':r['policy'],'methods':results})
    summary={};gates=[]
    for cohort in ['broad','small_pool']:
        summary[cohort]={}
        for policy in ['random','adaptive']:
            rr=[r for r in rows if r['cohort']==cohort and r['policy']==policy]
            methods={name:{'brier':np.mean([r['methods'][name]['brier'] for r in rr],axis=0).tolist(),
                           'alc':float(np.mean([r['methods'][name]['alc'] for r in rr])),
                           'alc_sample_sd':float(np.std([r['methods'][name]['alc'] for r in rr],ddof=1)),
                           'alc_p90':float(np.quantile([r['methods'][name]['alc'] for r in rr],.9)),
                           'alc_min':float(min(r['methods'][name]['alc'] for r in rr)),
                           'alc_max':float(max(r['methods'][name]['alc'] for r in rr))} for name in names}
            comparisons={name:paired([r['methods']['candidate']['alc']-r['methods'][name]['alc'] for r in rr],20261031) for name in ['full','no_rasch','em']}
            removals={name:paired([r['methods'][name]['alc']-r['methods']['candidate']['alc'] for r in rr],20261031) for name in CANDIDATE.ablations(config)}
            summary[cohort][policy]={'n_draws':len(rr),'methods':methods,'candidate_minus_reference':comparisons,'ablation_minus_candidate':removals}
            gates += [{'name':f'{cohort}/{policy}: mean improves {name}','passed':comparisons[name]['mean']<0} for name in ['full','no_rasch']]
            if policy=='adaptive':
                gates.append({'name':f'{cohort}: paired 95% upper below zero versus full','passed':comparisons['full']['percentile_95_interval'][1]<0})
                gates.append({'name':f'{cohort}: >=60% draw wins versus full','passed':comparisons['full']['wins']/len(rr)>=.6})
                gates += [{'name':f'{cohort}: removing {name} does not improve mean','passed':value['mean']>=-1e-12} for name,value in removals.items()]
    out=ROOT/'results/asrp_next/validation'
    result={'config':config,'cohorts':summary,'gates':gates,'passed':all(g['passed'] for g in gates),
        'locked_candidate_sha256':sha(lock_path),'validation_cache_index_sha256':sha(ROOT/'results/asrp_next/validation_cache/index.json'),
        'scoring_script_sha256':sha(Path(__file__)),
        'scope':'Held-out public groups for candidate selection; frozen training/calibration overlap remains. Small-pool draws are a composition stress test.'}
    write_json(out/'summary.json',result);write_json(out/'draws.json',rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
