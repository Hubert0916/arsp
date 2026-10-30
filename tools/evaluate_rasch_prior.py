"""Factorial removal of Rasch and the count-based prior at budgets zero and one."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import gzip
import io
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import statistics
import tempfile
import time

import evaluate_proxy as runtime
from prepare_proxy import ROOT, sha, split_group, write_json
from evaluate_ablation import read_jsonl, identity

VARIANTS = ('full', 'no_rasch', 'no_count_prior', 'no_count_prior_no_rasch')


@contextmanager
def condition(model, variant):
    resources = model._resources()
    prior, config = resources[2], resources[4]
    original_prior, lambdas = prior['feature_prior'], config['lambdas']
    try:
        if 'no_count_prior' in variant:
            prior['feature_prior'] = None
        if variant in ('no_rasch', 'no_count_prior_no_rasch'):
            config['lambdas'] = [0.] * 6
        yield
    finally:
        prior['feature_prior'], config['lambdas'] = original_prior, lambdas


def evaluate(task):
    from threadpoolctl import threadpool_limits
    group, traces = task
    model = runtime.MODEL
    runtime.VECTOR.cache_clear()
    runtime.DIFFICULTY.cache_clear()
    model.get_predictor().vectors.clear()
    model.get_predictor().fits.clear()
    records = []
    started = time.monotonic()
    with threadpool_limits(limits=2):
        for seed in runtime.CONFIG['seeds']:
            pool, targets = split_group(group, seed)
            by_id = {r['item_id']: r for r in pool}
            trace = traces[seed]
            chosen = [by_id[i] for i in trace['acquired_item_ids']]
            assert [r['label'] for r in chosen] == trace['labels']
            supports = [[[r['input'], r['label']] for r in chosen[:n]] for n in (0, 1)]
            rows = [{'source': group['source'], 'subject_id': group['subject_id'], 'seed': seed,
                     'policy': 'adaptive', 'item_id': r['item_id'], 'response_id': r['response_id'],
                     'y': r['label'], **{key: [] for v in VARIANTS for key in (v, v+'_raw')}} for r in targets]
            for j, n in enumerate((0, 1)):
                if model.PROBE_WEIGHTS[model.BUDGETS.index(n)] != 0:
                    raise ValueError('This experiment requires zero kernel weight')
                support = supports[j]
                for index, (row, target) in enumerate(zip(rows, targets)):
                    for variant in VARIANTS:
                        with condition(model, variant):
                            raw, vector = model._predict_hybrid(target['input'], support, return_vector=True)
                            p = model._calibration_adjust(raw, target['input'][1], vector, n,
                                                          model._resources()[2]['topic_calibration'])
                            row[variant].append(runtime.finite_probability(p))
                            row[variant+'_raw'].append(runtime.finite_probability(raw))
                            if index == 0:
                                native = model.predict(target['input'], support.copy())
                                if abs(native-p) > 1e-12:
                                    raise ValueError('Factorial value differs from native inference')
                # The item-only feature cache is shared; support/labels remain per seed.
            records.extend(rows)
    return group['source'], records, time.monotonic()-started


def score(directory):
    protocol = json.loads((ROOT/'experiments/rasch_prior_ablation.json').read_text())
    run = json.loads((directory/'run.json').read_text())
    if protocol != run['protocol']:
        raise ValueError('Protocol mismatch')
    for path, checksum in run['inputs'].items():
        if sha(ROOT/path) != checksum:
            raise ValueError('Input changed: '+path)
    if sha(directory/'predictions.jsonl.gz') != run['predictions_sha256']:
        raise ValueError('Output checksum mismatch')
    references = {identity(r): r for r in read_jsonl(ROOT/'results/ablation/predictions.jsonl.gz')}
    rows = read_jsonl(directory/'predictions.jsonl.gz')
    seen = set(); cells = defaultdict(list); maximum = 0.
    for row in rows:
        key = identity(row)
        if key in seen or key not in references:
            raise ValueError('Unexpected or duplicate item')
        seen.add(key); ref = references[key]
        if row['y'] != ref['y'] or row['response_id'] != ref['response_id']:
            raise ValueError('Changed evaluation response')
        for variant in VARIANTS:
            for field in (variant,variant+'_raw'):
                if len(row[field]) != 2:
                    raise ValueError('Missing budget')
                for p in row[field]:runtime.finite_probability(p)
        for variant in ('full','no_rasch'):
            maximum=max(maximum,max(abs(a-b) for a,b in zip(row[variant],ref[variant][:2])))
        maximum=max(maximum,max(abs(a-b) for a,b in zip(row['full_raw'],ref['no_topic'][:2])))
        cells[(row['source'],row['subject_id'],row['seed'])].append(row)
    if seen != set(references) or maximum > 1e-7:
        raise ValueError('Nominal conditions do not reproduce original ablations')
    source_rows=[]
    for (source,subject,seed),members in sorted(cells.items()):
        values={field:[math.fsum((r[field][j]-r['y'])**2 for r in members)/len(members) for j in range(2)]
                for v in VARIANTS for field in (v,v+'_raw')}
        source_rows.append({'source':source,'subject_id':subject,'seed':seed,'n_rows':len(members),**values})
    variants={}
    for v in VARIANTS:
        per_seed=[]
        for seed in protocol['seeds']:
            group_rows=[r for r in source_rows if r['seed']==seed]
            if len(group_rows)!=protocol['n_groups']:raise ValueError('Incomplete group coverage')
            per_seed.append({'seed':seed,'brier':[statistics.mean(r[v][j] for r in group_rows) for j in range(2)],
                             'raw_brier':[statistics.mean(r[v+'_raw'][j] for r in group_rows) for j in range(2)]})
        variants[v]={'brier':[statistics.mean(r['brier'][j] for r in per_seed) for j in range(2)],
                     'raw_brier':[statistics.mean(r['raw_brier'][j] for r in per_seed) for j in range(2)],'per_seed':per_seed}
    summary={'protocol':protocol,'scoring_script_sha256':sha(Path(__file__)),
             'n_prediction_records':len(rows),'max_nominal_prediction_difference':maximum,'variants':variants}
    write_json(directory/'summary.json',summary);write_json(directory/'by_source.json',source_rows)
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/rasch_prior_ablation')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--score-only',action='store_true')
    args=parser.parse_args()
    if args.score_only:
        score(args.output);return
    manifest=json.loads((ROOT/'experiments/proxy_manifest.json').read_text())
    protocol=json.loads((ROOT/'experiments/rasch_prior_ablation.json').read_text())
    if sha(ROOT/'data/proxy.jsonl') != manifest['prepared_sha256']:
        raise ValueError('Prepared data mismatch')
    frozen=json.loads((ROOT/'artifacts/manifest.sha256.json').read_text())
    for name,checksum in frozen.items():
        if sha(ROOT/name)!=checksum:raise ValueError('Frozen artifact mismatch')
    groups=read_jsonl(ROOT/'data/proxy.jsonl');traces=defaultdict(dict)
    for t in read_jsonl(ROOT/'results/proxy/acquisition.jsonl'):
        if t['policy']=='adaptive':traces[(t['source'],t['subject_id'])][t['seed']]=t
    args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='asrp-prior-ablation-') as temp:
        bundle=Path(temp)/'bundle';chunks=Path(temp)/'chunks';bundle.mkdir();chunks.mkdir()
        for name in frozen:
            relative=Path(name).relative_to('src/asrp' if name.startswith('src/') else 'artifacts/asrp')
            target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=runtime.init_worker,initargs=(bundle,manifest['config'])) as executor:
            futures=[executor.submit(evaluate,(g,traces[(g['source'],g['subject_id'])])) for g in groups]
            for index,future in enumerate(as_completed(futures),1):
                source,records,elapsed=future.result()
                (chunks/f'{source}.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))
                print(f'Prior ablation {index}/{len(groups)}: {source} ({elapsed:.1f}s)',flush=True)
        with (args.output/'predictions.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
                with io.TextIOWrapper(zipped,encoding='utf-8') as stream:
                    for g in groups:stream.write((chunks/f"{g['source']}.jsonl").read_text())
    inputs=['experiments/proxy_manifest.json','experiments/rasch_prior_ablation.json','artifacts/manifest.sha256.json',
            'uv.lock','results/proxy/acquisition.jsonl','results/ablation/predictions.jsonl.gz']
    write_json(args.output/'run.json',{'protocol':protocol,'prepared_data_sha256':manifest['prepared_sha256'],
        'inputs':{name:sha(ROOT/name) for name in inputs},'script_sha256':sha(Path(__file__)),
        'predictions_sha256':sha(args.output/'predictions.jsonl.gz')})
    score(args.output)


if __name__=='__main__':main()
