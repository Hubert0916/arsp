"""Isolate the effect of other subjects' labels in the same benchmark.

Use the first three source/benchmark names (lexical order) with two released
groups fitting the 1,000-pair cap and distinct visible subjects. Keep the focal
subject's queries and acquired labels fixed; only append companion evidence.
This is a mechanism diagnostic, not another estimate of competition performance.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import itertools
import json
import multiprocessing
import shutil
import tempfile
from pathlib import Path

import evaluate_proxy as runtime
from draw_protocol import BUDGETS, WEIGHTS, fixed_split, input_pair
from prepare_proxy import ROOT, sha, write_json


def check(task):
    from threadpoolctl import threadpool_limits
    focal,companion,evidence_a,evidence_b,seed=task
    model=runtime.MODEL
    _,targets=fixed_split(focal,seed)
    curves=[[],[]];max_errors=[];baseline_differences=[]
    with threadpool_limits(limits=2):
        for budget in BUDGETS:
            caches=[{},{}];losses=[[],[]];differences=[]
            for item in targets:
                for response in item['responses']:
                    x=input_pair(focal,item,response)
                    key=json.dumps(x,sort_keys=True)
                    pp=[]
                    for variant,evidence in enumerate([evidence_a[:budget],evidence_a[:budget]+evidence_b[:budget]]):
                        if key not in caches[variant]:
                            caches[variant][key]=runtime.finite_probability(model.predict(x,evidence))
                        p=caches[variant][key];pp.append(p)
                        losses[variant].append((p-response['label'])**2)
                    differences.append(abs(pp[0]-pp[1]))
            for j in range(2):curves[j].append(sum(losses[j])/len(losses[j]))
            max_errors.append(max(differences))
            a=runtime.BASELINE.predict(x,evidence_a[:budget])
            b=runtime.BASELINE.predict(x,evidence_a[:budget]+evidence_b[:budget])
            baseline_differences.append(abs(a-b))
    assert max(max_errors[:2])<1e-12 and max(baseline_differences)==0
    return {'source':focal['source'],'focal_group':focal['group_id'],'companion_group':companion['group_id'],
            'n_source_pairs':focal['n_items']+companion['n_items'],
            'n_evaluation_responses':sum(len(i['responses']) for i in targets),
            'focal_only_brier':curves[0],'with_companion_brier':curves[1],
            'brier_delta':[b-a for a,b in zip(*curves)],
            'max_prediction_change':max_errors,
            'empirical_mean_max_change':max(baseline_differences),
            'focal_only_alc':sum(w*v for w,v in zip(WEIGHTS,curves[0])),
            'with_companion_alc':sum(w*v for w,v in zip(WEIGHTS,curves[1]))}


def main():
    manifest=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    assert sha(ROOT/'data/formative.jsonl')==manifest['prepared_sha256']
    with (ROOT/'data/formative.jsonl').open() as f:groups={g['group_id']:g for g in map(json.loads,f)}
    candidates=defaultdict(list)
    for g in groups.values():candidates[(g['source'],g['benchmark_id'])].append(g)
    pairs=[]
    for key,gg in sorted(candidates.items()):
        feasible=[(a,b) for a,b in itertools.combinations(sorted(gg,key=lambda g:g['group_id']),2)
                  if a['n_items']+b['n_items']<=1000 and a['subject']!=b['subject']]
        if feasible:pairs.append(feasible[0])
        if len(pairs)==3:break
    wanted={g['group_id'] for pair in pairs for g in pair};supports={}
    with gzip.open(ROOT/'results/formative/acquisition.jsonl.gz','rt') as f:
        for t in map(json.loads,f):
            if t['policy']!='adaptive':continue
            for gid in t['groups']:
                if gid not in wanted or gid in supports:continue
                g=groups[gid];items={i['item_id']:i for i in g['items']};rows=[]
                for d in t['decisions']:
                    if d['group_id']!=gid or not d['acquired']:continue
                    item=items[d['item_id']];r=next(r for r in item['responses'] if r['response_id']==d['response_id'])
                    rows.append([input_pair(g,item,r),r['label']])
                assert len(rows)==31
                supports[gid]=rows
    tasks=[(a,b,supports[a['group_id']],supports[b['group_id']],manifest['config']['snapshot_seed']) for a,b in pairs]
    frozen=json.loads((ROOT/'artifacts/manifest.sha256.json').read_text())
    with tempfile.TemporaryDirectory(prefix='asrp-context-audit-') as td:
        bundle=Path(td)
        for name,h in frozen.items():
            assert sha(ROOT/name)==h
            prefix='src/asrp' if name.startswith('src/') else 'artifacts/asrp'
            target=bundle/Path(name).relative_to(prefix);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        rows=[]
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),
                initializer=runtime.init_worker,initargs=(bundle,manifest['config'])) as pool:
            for f in as_completed([pool.submit(check,t) for t in tasks]):
                r=f.result();rows.append(r);print(json.dumps(r),flush=True)
    write_json(ROOT/'results/rasch_audit/context_check.json',{
        'selection':'First three source/benchmark names in lexical order, first group-ID pair fitting 1,000 pairs; distinct visible subjects; no score selection.',
        'intervention':'Add companion support only; focal labels, targets, weights, calibration, and acquired-prefix order fixed. Not a rerun of joint acquisition.',
        'budgets':list(BUDGETS),'results':sorted(rows,key=lambda r:r['source']),
        'input_manifest_sha256':sha(ROOT/'experiments/formative_manifest.json'),
        'inference_manifest_sha256':sha(ROOT/'artifacts/manifest.sha256.json'),
        'script_sha256':sha(Path(__file__))})


if __name__=='__main__':main()
