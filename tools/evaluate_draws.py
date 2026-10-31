"""Run complete formative-style draws with global evidence and repeated responses."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time

import evaluate_proxy as runtime
from evaluate_ablation import component_predictions, disabled
from prepare_proxy import ROOT, sha, write_json
from draw_protocol import BUDGETS, fixed_split, input_pair, random_choose, trajectory

METHODS=('full','no_rasch','no_kernel','no_topic','empirical_mean')


def init_worker(bundle, config, data):
    global GROUPS
    runtime.init_worker(bundle,config)
    with open(data) as f:GROUPS={g['group_id']:g for g in map(json.loads,f)}


def evaluate_draw(draw):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):return evaluate_draw_inner(draw)


def evaluate_draw_inner(draw):
    start=time.monotonic();model=runtime.MODEL;cfg=runtime.CONFIG
    groups=[GROUPS[gid] for gid in draw['group_ids']]
    predictions,traces=[],[]
    runtime.VECTOR.cache_clear();runtime.DIFFICULTY.cache_clear()
    model.get_predictor().vectors.clear();model.get_predictor().fits.clear()
    for policy in cfg['policies']:
        if policy != 'random':
            raise ValueError('The current release supports only random acquisition')
        records=[];targets=[]
        for g in groups:
            _,evaluation=fixed_split(g,cfg['snapshot_seed'])
            for item in evaluation:
                for response in item['responses']:
                    records.append({'draw_id':draw['draw_id'],'group_id':g['group_id'],
                        'source':g['source'],'benchmark_id':g['benchmark_id'],'subject_id':g['subject_id'],
                        'policy':policy,'item_id':item['item_id'],'response_id':response['response_id'],
                        'y':response['label'],**{v:[] for v in METHODS}})
                    targets.append(input_pair(g,item,response))
        def choose(x,evidence,context):
            decision=runtime.LABELING.acquisition_function(x,labeled=evidence,context=context)
            if decision != random_choose(x,context):
                raise ValueError('Submission hook differs from uniform random acquisition')
            return decision
        def checkpoint(budget,evidence,counts):
            if any(n!=budget for n in counts):raise ValueError('Policy failed to reach checkpoint')
            model.get_predictor().fits.clear()
            cache={};validated=set()
            for record,x in zip(records,targets):
                key=json.dumps(x,sort_keys=True,separators=(',',':'))
                if key not in cache:
                    actual_n=sum(s==x[0] and i.get('benchmark_id')==x[1]['benchmark_id'] for (s,i),y in evidence)
                    if actual_n==budget:
                        values=component_predictions(model,x,evidence,budget)
                    else:
                        # Preserve native behavior if public subjects have
                        # indistinguishable visible dictionaries.
                        values={}
                        for variant in METHODS[:-1]:
                            with disabled(model,variant):values[variant]=runtime.finite_probability(model.predict(x,evidence))
                    values['empirical_mean']=runtime.finite_probability(runtime.BASELINE.predict(x,evidence))
                    cache[key]=values
                values=cache[key]
                if record['group_id'] not in validated:
                    # Every group/checkpoint/policy/ablation is checked against
                    # the packaged path, using all groups' revealed evidence.
                    for variant in METHODS[:-1]:
                        with disabled(model,variant):native=model.predict(x,evidence.copy())
                        if abs(native-values[variant])>1e-12:raise ValueError('Native inference mismatch')
                    validated.add(record['group_id'])
                for variant,value in values.items():record[variant].append(value)
        decisions,counts=trajectory(groups,cfg['snapshot_seed'],choose,checkpoint)
        predictions.extend(records)
        traces.append({'draw_id':draw['draw_id'],'policy':policy,'groups':draw['group_ids'],
            'n_source_pairs':draw['n_source_pairs'],'counts':counts,'decisions':decisions})
    return draw['draw_id'],predictions,traces,time.monotonic()-start


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/formative')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    manifest=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    config=json.loads((ROOT/'experiments/formative_config.json').read_text())
    if manifest['config']!=config:raise ValueError('Config mismatch')
    data=ROOT/'data/formative.jsonl'
    if sha(data)!=manifest['prepared_sha256']:raise ValueError('Prepared data checksum mismatch')
    frozen=json.loads((ROOT/'artifacts/manifest.sha256.json').read_text())
    for name,checksum in frozen.items():
        if sha(ROOT/name)!=checksum:raise ValueError('Frozen inference changed')
    paths=['experiments/formative_config.json','experiments/formative_manifest.json','artifacts/manifest.sha256.json','uv.lock']
    scripts=['tools/evaluate_draws.py','tools/draw_protocol.py','tools/prepare_draws.py','tools/evaluate_ablation.py','tools/evaluate_proxy.py']
    input_hashes={name:sha(ROOT/name) for name in paths}
    script_hashes={name:sha(ROOT/name) for name in scripts}
    args.output.mkdir(parents=True,exist_ok=True)
    chunks=args.output/'chunks';chunks.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='asrp-formative-') as temp:
        bundle=Path(temp)
        for name in frozen:
            prefix='src/asrp' if name.startswith('src/') else 'artifacts/asrp'
            target=bundle/Path(name).relative_to(prefix)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=init_worker,initargs=(bundle,config,data)) as pool:
            futures=[pool.submit(evaluate_draw,d) for d in manifest['draws']]
            for future in as_completed(futures):
                did,records,traces,seconds=future.result()
                for kind,rows in [('predictions',records),('acquisition',traces)]:
                    (chunks/f'{did:03d}.{kind}.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
                print(f'Draw {did}: {len(records)} scored rows under random acquisition; {seconds:.1f}s',flush=True)
    for kind in ('predictions','acquisition'):
        with (args.output/f'{kind}.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
                with io.TextIOWrapper(compressed,encoding='utf-8') as stream:
                    for d in manifest['draws']:stream.write((chunks/f"{d['draw_id']:03d}.{kind}.jsonl").read_text())
    for name,checksum in {**input_hashes,**script_hashes,**frozen}.items():
        if sha(ROOT/name)!=checksum:raise ValueError('File changed during evaluation: '+name)
    write_json(args.output/'run.json',{'protocol':config,'prepared_sha256':sha(data),
        'inputs':input_hashes,
        'scripts':script_hashes,
        'files':{f'{kind}.jsonl.gz':sha(args.output/f'{kind}.jsonl.gz') for kind in ('predictions','acquisition')}})


if __name__=='__main__':main()
