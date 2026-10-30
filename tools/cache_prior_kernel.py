"""Cache frozen prediction components for no-Rasch development/validation.

All repeated evaluation outcomes are retained via binary sufficient statistics.
No evaluation outcomes enter prediction or acquisition; they are used for scoring.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

import evaluate_proxy as runtime
from draw_protocol import BUDGETS, fixed_split, acquired_response, input_pair, random_choose, trajectory
from prepare_proxy import ROOT, sha, write_json


def compact(group, seed):
    acquisition,_=fixed_split(group,seed)
    ids={i['item_id'] for i in acquisition}
    items=[]
    for item in group['items']:
        if item['item_id'] in ids:
            responses=[dict(acquired_response(group,item,seed),multiplicity=1,successes=acquired_response(group,item,seed)['label'])]
        else:
            unique={}
            for response in item['responses']:
                key=response['interactors']
                if key not in unique:unique[key]=dict(response,multiplicity=0,successes=0)
                unique[key]['multiplicity']+=1;unique[key]['successes']+=response['label']
            responses=list(unique.values())
        items.append(dict(item,responses=responses))
    return dict(group,items=items)


def initialize(bundle,config):
    runtime.init_worker(bundle,config)


def run_draw(task):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):return run_inner(task)


def run_inner(task):
    from base_features import subject_key
    from project_core import transform
    draw,groups=task;model=runtime.MODEL;cfg=runtime.CONFIG;started=time.monotonic()
    runtime.VECTOR.cache_clear();runtime.DIFFICULTY.cache_clear()
    predictor=model.get_predictor();predictor.vectors.clear();predictor.fits.clear()
    priors=model._resources()[2]
    prior_cache={}
    def prior(x):
        key=json.dumps([x[0],x[1].get('item_features','')],sort_keys=True)
        if key not in prior_cache:prior_cache[key]=model._feature_prior(*x,priors['feature_prior'])
        return prior_cache[key]
    records=[];traces=[]
    for policy in cfg['policies']:
        runtime.LABELING._policy=None
        targets=[];identity=[];count=[];success=[]
        for gid,g in enumerate(groups):
            _,items=fixed_split(g,cfg['snapshot_seed'])
            for item in items:
                for r in item['responses']:
                    targets.append(input_pair(g,item,r));identity.append(gid)
                    count.append(r['multiplicity']);success.append(r['successes'])
        nrows=len(targets)
        arrays={k:np.zeros((nrows,6)) for k in ['full','no_rasch','em','kernel_base','residual','topic','prior','support_prior','n','s']}
        def choose(x,evidence,context):
            return random_choose(x,context) if policy=='random' else runtime.LABELING.acquisition_function(x,labeled=evidence,context=context)
        def checkpoint(budget,evidence,counts):
            assert all(n==budget for n in counts)
            j=BUDGETS.index(budget);seen={};subject_stats={}
            for row,x in enumerate(targets):
                key=json.dumps(x,sort_keys=True,separators=(',',':'))
                if key not in seen:
                    sk=(subject_key(x[0]),x[1]['benchmark_id'])
                    if sk not in subject_stats:
                        own=[(z,y) for z,y in evidence if z[0]==x[0] and z[1].get('benchmark_id')==x[1]['benchmark_id']][:31]
                        subject_stats[sk]=(len(own),sum(y for _,y in own),np.mean([prior(z) for z,_ in own]) if own else .5)
                    n,s,u=subject_stats[sk];q=prior(x)
                    hybrid,vector=model._predict_hybrid(x,evidence,return_vector=True)
                    pk=float(predictor.predict(x,evidence));base=predictor.initial(x[0]);residual=0.
                    if n:
                        positions,state=next(reversed(predictor.fits.values()))
                        pos=positions[subject_key(x[0])];base=float(state['base'][pos])
                        features=predictor.features([x[1]])
                        semantic,lexical=transform(state['geometry'],features[:,:384],features[:,384:])
                        raw=(semantic @ state['semantic'].T+lexical @ state['lexical'].T)/2
                        k=np.exp(np.minimum(raw-1,0)/state['bandwidth'])
                        residual=float((k @ state['dual'])[0]-state['bias'][pos])
                    assert abs(np.clip(base+residual,0,1)-pk)<1e-12
                    delta=model._calibration_adjust(.5,x[1],vector,n,priors['topic_calibration'])-.5
                    w=float(np.interp(n,BUDGETS,model.PROBE_WEIGHTS))
                    smooth=(s+4*q)/(n+4)
                    vals={'full':float(np.clip(w*pk+(1-w)*hybrid+delta,0,1)),
                          'no_rasch':float(np.clip(w*pk+(1-w)*smooth+delta,0,1)),
                          'em':runtime.finite_probability(runtime.BASELINE.predict(x,evidence)),
                          'kernel_base':base,'residual':residual,'topic':delta,'prior':q,
                          'support_prior':u,'n':n,'s':s}
                    seen[key]=vals
                for k,v in seen[key].items():arrays[k][row,j]=v
            # Native prediction checks at every group and checkpoint.
            for gid in range(len(groups)):
                row=identity.index(gid)
                assert abs(model.predict(targets[row],evidence)-arrays['full'][row,j])<1e-12
        decisions,counts=trajectory(groups,cfg['snapshot_seed'],choose,checkpoint)
        records.append({'draw_id':draw['draw_id'],'policy':policy,'groups':[g['group_id'] for g in groups],
                        'arrays':dict(arrays,count=np.array(count),success=np.array(success),group=np.array(identity))})
        traces.append({'draw_id':draw['draw_id'],'policy':policy,'groups':draw['group_ids'],'decisions':decisions,'counts':counts})
    return records,traces,time.monotonic()-started


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'experiments/formative_manifest.json')
    p.add_argument('--data',type=Path,default=ROOT/'data/formative.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'results/asrp_next/development_cache')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();manifest=json.loads(args.manifest.read_text());config=manifest['config']
    assert sha(args.data)==manifest['prepared_sha256']
    groups={}
    with args.data.open() as f:
        for line in f:
            g=compact(json.loads(line),config['snapshot_seed']);groups[g['group_id']]=g
    args.output.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'artifacts/manifest.sha256.json').read_text())
    index=[];traces=[]
    with tempfile.TemporaryDirectory(prefix='asrp-components-') as td:
        bundle=Path(td)
        for name,digest in frozen.items():
            assert sha(ROOT/name)==digest
            prefix='src/asrp' if name.startswith('src/') else 'artifacts/asrp'
            target=bundle/Path(name).relative_to(prefix);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(bundle,config)) as pool:
            futures=[pool.submit(run_draw,(d,[groups[g] for g in d['group_ids']])) for d in manifest['draws']]
            for i,f in enumerate(as_completed(futures),1):
                records,tt,elapsed=f.result();traces.extend(tt)
                for row in records:
                    arrays=row.pop('arrays');filename=f"{row['draw_id']:03d}_{row['policy']}.npz"
                    np.savez_compressed(args.output/filename,**arrays)
                    index.append(dict(row,file=filename,sha256=sha(args.output/filename)))
                print(f'Cached {i}/{len(futures)} draws ({elapsed:.1f}s)',flush=True)
    write_json(args.output/'index.json',{'manifest_sha256':sha(args.manifest),'data_sha256':sha(args.data),
        'inference_manifest_sha256':sha(ROOT/'artifacts/manifest.sha256.json'),
        'script_sha256':sha(Path(__file__)),'draws':sorted(index,key=lambda r:(r['draw_id'],r['policy']))})
    write_json(args.output/'acquisition.json',sorted(traces,key=lambda r:(r['draw_id'],r['policy'])))


if __name__=='__main__':main()
