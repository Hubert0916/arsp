"""Develop one global no-Rasch configuration; validation is a separate command."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

from prepare_proxy import ROOT,sha,write_json

SPEC=importlib.util.spec_from_file_location('next_candidate',ROOT/'experiments/asrp_next/prior_kernel.py')
CANDIDATE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(CANDIDATE)
ALC=np.array([.1,.2,.2,.2,.2,.1])


def load_cache(directory):
    index=json.loads((directory/'index.json').read_text());cache=[]
    for r in index['draws']:
        assert sha(directory/r['file'])==r['sha256']
        arrays=dict(np.load(directory/r['file']))
        ng=len(r['groups']);gid=arrays['group'];weight=np.zeros(len(gid))
        for g in range(ng):
            rows=gid==g
            weight[rows]=1/(ng*arrays['count'][rows].sum())
        arrays['row_weight']=weight
        cache.append((r,arrays))
    return cache,index


def curve(parts,p):
    return np.sum(parts['row_weight'][:,None]*(parts['count'][:,None]*p*p-2*parts['success'][:,None]*p+parts['success'][:,None]),axis=0)


def evaluate(cache,config):
    rows=[]
    for metadata,parts in cache:
        b=curve(parts,CANDIDATE.probabilities(parts,config))
        rows.append({'draw_id':metadata['draw_id'],'policy':metadata['policy'],'brier':b.tolist(),'alc':float(b@ALC)})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=ROOT/'results/asrp_next/development_cache');args=p.parse_args()
    assert args.directory.name=='development_cache','This search must not access validation.'
    cache,index=load_cache(args.directory)
    reference=json.loads((ROOT/'results/formative/draws.json').read_text())
    reference={(r['draw_id'],r['policy']):r for r in reference}
    max_error=0.
    plain_error=0.
    plain_config={'prior_shrink':1.,'prior_center':.5,'concentrations':[4.],
                  'masses':[1.],'residual_scale':1.,'combine':'blend','topic_decay':None}
    for r,parts in cache:
        for field,variant in [('full','full'),('no_rasch','no_rasch'),('em','empirical_mean')]:
            b=curve(parts,parts[field]);expected=reference[(r['draw_id'],r['policy'])]['methods'][variant]['brier']
            max_error=max(max_error,float(np.max(abs(b-expected))))
        plain_error=max(plain_error,float(np.max(abs(CANDIDATE.probabilities(parts,plain_config)-parts['no_rasch']))))
    assert max_error<1e-10,('Component cache does not reproduce reference',max_error)
    assert plain_error<1e-12,('Candidate family does not reproduce plain no-Rasch',plain_error)
    choices=[]
    development_rate=float(np.mean([np.sum(parts['row_weight']*parts['success']) for _,parts in cache]))
    distributions=[([2.],[1.]),([4.],[1.]),([8.],[1.]),([1.,8.],[.3,.7]),([1.,16.],[.3,.7]),([2.,8.],[.3,.7])]
    for alpha,center,(cs,ms),scale,combine,decay in itertools.product([.5,.75,1.],[.5,development_rate],distributions,[.25,.5,1.],['rebase','blend'],[None,8.]):
        if alpha==1. and center!=.5:continue
        config={'prior_shrink':alpha,'prior_center':center,'concentrations':cs,'masses':ms,'residual_scale':scale,'combine':combine,'topic_decay':decay}
        rows=evaluate(cache,config)
        scores={policy:float(np.mean([r['alc'] for r in rows if r['policy']==policy])) for policy in ['random','adaptive']}
        choices.append({'config':config,'mean_alc':float(np.mean(list(scores.values()))),'policy_alc':scores})
        if len(choices)%36==0:print(f'Screened {len(choices)} global configurations on development only',flush=True)
    choices.sort(key=lambda r:r['mean_alc'])
    eligible=[]
    for candidate in choices[:30]:
        config=candidate['config'];deltas={}
        for name in CANDIDATE.ablations(config):
            scores=[]
            for r,parts in cache:
                if r['policy']=='adaptive':scores.append(float(curve(parts,CANDIDATE.probabilities(parts,config,name))@ALC))
            deltas[name]=float(np.mean(scores)-candidate['policy_alc']['adaptive'])
        candidate['development_ablation_deltas']=deltas
        if all(d>=0 for d in deltas.values()):eligible.append(candidate)
    best=(eligible or choices)[0]
    directory=ROOT/'results/asrp_next/development'
    write_json(directory/'search.json',{'n_configurations':len(choices),'development_equal_draw_success_rate':development_rate,'cache_max_reference_brier_error':max_error,
        'plain_no_rasch_max_probability_error':plain_error,
        'selection':'Lowest equal-policy/equal-draw development ALC among top-30 configurations whose listed component removals do not improve Adaptive mean ALC; otherwise best ALC.',
        'selected':best,'ranking':choices,'development_cache_index_sha256':sha(args.directory/'index.json'),
        'candidate_implementation_sha256':sha(ROOT/'experiments/asrp_next/prior_kernel.py'),'search_script_sha256':sha(Path(__file__))})
    write_json(ROOT/'experiments/asrp_next/locked_candidate.json',{'config':best['config'],
        'selected_using':'Development only; no validation scores inspected.',
        'search_result_sha256':sha(directory/'search.json'),
        'implementation_sha256':sha(ROOT/'experiments/asrp_next/prior_kernel.py'),
        'protocol_sha256':sha(ROOT/'experiments/asrp_next/protocol.json')})
    print(json.dumps(best,indent=2))


if __name__=='__main__':main()
