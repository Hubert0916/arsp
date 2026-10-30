"""Reproduce component removals with the published adaptive evidence held fixed.

Run after preparing data/proxy.jsonl. Scoring and reporting need only the released
compressed predictions: use --score-only to skip inference.
"""
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
import statistics
import tempfile
import time

import evaluate_proxy as runtime
from prepare_proxy import ROOT, sha, split_group, write_json

VARIANTS = ('full', 'no_rasch', 'no_kernel', 'no_topic')
LABELS = {'full': 'Full ASRP', 'no_rasch': 'Without Rasch',
          'no_kernel': 'Without Kernel Probe', 'no_topic': 'Without topic calibration'}


@contextmanager
def disabled(model, variant):
    """Temporarily remove one component from the native runtime."""
    config = model._resources()[4]
    lambdas, weights, calibrate = config['lambdas'], model.PROBE_WEIGHTS, model._calibration_adjust
    try:
        if variant == 'no_rasch':
            config['lambdas'] = [0.] * len(model.BUDGETS)
        elif variant == 'no_kernel':
            model.PROBE_WEIGHTS = (0.,) * len(model.BUDGETS)
        elif variant == 'no_topic':
            model._calibration_adjust = lambda p, *args: p
        yield
    finally:
        config['lambdas'], model.PROBE_WEIGHTS, model._calibration_adjust = lambdas, weights, calibrate


def component_predictions(model, item_input, support, budget):
    """Evaluate shared branches once; use the native correction for each variant."""
    hybrid, vector = model._predict_hybrid(item_input, support, return_vector=True)
    kernel = float(model.get_predictor().predict(item_input, support))
    with disabled(model, 'no_rasch'):
        smooth = model._predict_hybrid(item_input, support)
    weight = model.PROBE_WEIGHTS[model.BUDGETS.index(budget)]
    prior = model._resources()[2]['topic_calibration']
    raw = weight * kernel + (1. - weight) * hybrid
    def corrected(p):
        return runtime.finite_probability(model._calibration_adjust(p, item_input[1], vector, budget, prior))
    values = {'full': corrected(raw), 'no_rasch': corrected(weight * kernel + (1. - weight) * smooth),
              'no_kernel': corrected(hybrid), 'no_topic': runtime.finite_probability(raw)}
    if weight == 0 and values['no_kernel'] != values['full']:
        raise ValueError('Removing an unused kernel changed the prediction')
    if weight == 1 and values['no_rasch'] != values['full']:
        raise ValueError('Removing an unused Rasch branch changed the prediction')
    return values


def evaluate_group(task):
    from threadpoolctl import threadpool_limits
    group, trajectories = task
    model = runtime.MODEL
    started = time.monotonic()
    model.get_predictor().vectors.clear()
    model.get_predictor().fits.clear()
    runtime.VECTOR.cache_clear()
    runtime.DIFFICULTY.cache_clear()
    records = []
    with threadpool_limits(limits=2):
        for seed in runtime.CONFIG['seeds']:
            pool, targets = split_group(group, seed)
            trace = trajectories[str(seed)]
            items = {r['item_id']: r for r in pool}
            ids = trace['acquired_item_ids']
            if len(ids) != 31 or len(set(ids)) != 31 or any(i not in items for i in ids):
                raise ValueError('Invalid acquired-item prefix')
            if [items[i]['label'] for i in ids] != trace['labels']:
                raise ValueError('Acquired labels do not match prepared data')
            acquired = [[items[i]['input'], items[i]['label']] for i in ids]
            rows = [{'source': group['source'], 'subject_id': group['subject_id'], 'seed': seed,
                     'policy': 'adaptive', 'item_id': r['item_id'], 'response_id': r['response_id'],
                     'y': r['label'], **{v: [] for v in VARIANTS}} for r in targets]
            model.get_predictor().fits.clear()
            for budget in runtime.CONFIG['budgets']:
                support = acquired[:budget]
                for index, (row, target) in enumerate(zip(rows, targets)):
                    values = component_predictions(model, target['input'], support, budget)
                    if index == 0:
                        # Verify each removal against the actual native prediction path.
                        for variant in VARIANTS:
                            with disabled(model, variant):
                                native = runtime.finite_probability(model.predict(target['input'], support.copy()))
                            if abs(native - values[variant]) > 1e-12:
                                raise ValueError(f'Component calculation differs from native {variant}')
                    for variant, value in values.items():
                        row[variant].append(value)
            records.extend(rows)
    return group['source'], records, time.monotonic() - started


def read_jsonl(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def identity(row):
    return tuple(row[k] for k in ('source', 'subject_id', 'seed', 'policy', 'item_id'))


def score(directory):
    manifest = json.loads((ROOT / 'experiments/proxy_manifest.json').read_text())
    config = manifest['config']
    run = json.loads((directory / 'run.json').read_text())
    protocol = json.loads((ROOT / 'experiments/ablation_config.json').read_text())
    if run['protocol'] != protocol or run['n_groups'] != len(manifest['groups']):
        raise ValueError('Ablations must cover the complete fixed protocol')
    for name, checksum in run['inputs'].items():
        if sha(ROOT / name) != checksum:
            raise ValueError(f'Ablation input differs: {name}')
    if run['inputs']['experiments/proxy_manifest.json'] != sha(ROOT / 'experiments/proxy_manifest.json'):
        raise ValueError('Split manifest differs')
    if sha(directory / 'predictions.jsonl.gz') != run['predictions_sha256']:
        raise ValueError('Ablation predictions differ from the run checksum')
    references = {identity(r): r for r in read_jsonl(ROOT / 'results/proxy/predictions.jsonl')
                  if r['policy'] == 'adaptive'}
    records = read_jsonl(directory / 'predictions.jsonl.gz')
    cells = defaultdict(list)
    seen, maximum_error = set(), 0.
    for row in records:
        key = identity(row)
        if key in seen or key not in references:
            raise ValueError('Duplicate or unexpected ablation evaluation item')
        seen.add(key)
        ref = references[key]
        if row['y'] != ref['y'] or row['response_id'] != ref['response_id']:
            raise ValueError('Evaluation response changed')
        for variant in VARIANTS:
            if len(row[variant]) != len(config['budgets']):
                raise ValueError('Missing budget prediction')
            for p in row[variant]:
                runtime.finite_probability(p)
        maximum_error = max(maximum_error, max(abs(p-q) for p, q in zip(row['full'], ref['asrp'])))
        cells[(row['source'], row['subject_id'], row['seed'])].append(row)
    if seen != set(references) or maximum_error > 1e-7:
        raise ValueError('Full ASRP does not reproduce all published adaptive predictions')
    expected_cells = {(g['source'], g['subject_id'], s) for g in manifest['groups'] for s in config['seeds']}
    if set(cells) != expected_cells:
        raise ValueError('Incomplete source/seed coverage')
    by_group = []
    for (source, subject, seed), rows in sorted(cells.items()):
        for variant in VARIANTS:
            curve = [math.fsum((r[variant][j]-r['y'])**2 for r in rows)/len(rows) for j in range(6)]
            by_group.append({'source': source, 'subject_id': subject, 'seed': seed, 'variant': variant,
                             'n_rows': len(rows), 'brier': curve,
                             'alc': math.fsum(w*b for w, b in zip(config['alc_weights'], curve))})
    variants = {}
    for variant in VARIANTS:
        per_seed = []
        for seed in config['seeds']:
            rows = [r for r in by_group if r['variant'] == variant and r['seed'] == seed]
            per_seed.append({'seed': seed, 'alc': statistics.mean(r['alc'] for r in rows),
                             'brier': [statistics.mean(r['brier'][j] for r in rows) for j in range(6)]})
        variants[variant] = {'brier': [statistics.mean(r['brier'][j] for r in per_seed) for j in range(6)],
                             'alc': statistics.mean(r['alc'] for r in per_seed),
                             'alc_seed_sd': statistics.stdev(r['alc'] for r in per_seed), 'per_seed': per_seed}
    for value in variants.values():
        value['delta_alc_vs_full'] = value['alc'] - variants['full']['alc']
    summary = {'protocol': protocol, 'n_groups': len(manifest['groups']), 'n_prediction_records': len(records),
               'max_full_prediction_difference': maximum_error, 'variants': variants}
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'by_source.json', by_group)
    print(json.dumps(summary, indent=2))
    return summary


def run_inference(args):
    manifest_path = ROOT / 'experiments/proxy_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    config = manifest['config']
    protocol = json.loads((ROOT / 'experiments/ablation_config.json').read_text())
    if protocol['seeds'] != config['seeds'] or protocol['budgets'] != config['budgets']:
        raise ValueError('Ablation and primary protocol differ')
    if sha(ROOT / 'data/proxy.jsonl') != manifest['prepared_sha256']:
        raise ValueError('Prepare the pinned proxy data before running ablations')
    reference_run = json.loads((ROOT / 'results/proxy/run.json').read_text())
    for name, checksum in reference_run['files'].items():
        if sha(ROOT / 'results/proxy' / name) != checksum:
            raise ValueError('Published reference checksum mismatch')
    frozen = json.loads((ROOT / 'artifacts/manifest.sha256.json').read_text())
    for name, checksum in frozen.items():
        if sha(ROOT / name) != checksum:
            raise ValueError(f'Inference artifact mismatch: {name}')
    groups = read_jsonl(ROOT / 'data/proxy.jsonl')
    traces = defaultdict(dict)
    for row in read_jsonl(ROOT / 'results/proxy/acquisition.jsonl'):
        if row['policy'] == 'adaptive':
            key = (row['source'], row['subject_id'])
            if str(row['seed']) in traces[key]:
                raise ValueError('Duplicate acquisition trajectory')
            traces[key][str(row['seed'])] = row
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='asrp-ablation-') as temp:
        bundle, chunks = Path(temp) / 'bundle', Path(temp) / 'chunks'
        bundle.mkdir(); chunks.mkdir()
        # Assemble exactly the verified inference files, without editing the submission.
        import shutil
        for name in frozen:
            relative = Path(name).relative_to('src/asrp' if name.startswith('src/') else 'artifacts/asrp')
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=runtime.init_worker, initargs=(bundle, config)) as executor:
            futures = [executor.submit(evaluate_group, (g, traces[(g['source'], g['subject_id'])])) for g in groups]
            for number, future in enumerate(as_completed(futures), 1):
                source, records, elapsed = future.result()
                (chunks / f'{source}.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in records))
                print(f'Ablated {number}/{len(groups)}: {source} ({elapsed:.1f}s)', flush=True)
        with (args.output / 'predictions.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding='utf-8') as stream:
                    for g in groups:
                        stream.write((chunks / f"{g['source']}.jsonl").read_text())
    inputs = ['experiments/proxy_manifest.json', 'experiments/ablation_config.json',
              'artifacts/manifest.sha256.json', 'uv.lock', 'results/proxy/predictions.jsonl',
              'results/proxy/acquisition.jsonl']
    write_json(args.output / 'run.json', {'protocol': protocol, 'n_groups': len(groups),
               'prepared_data_sha256': manifest['prepared_sha256'],
               'inputs': {name: sha(ROOT / name) for name in inputs},
               'scripts': {name: sha(ROOT / name) for name in ['tools/evaluate_ablation.py', 'tools/evaluate_proxy.py']},
               'predictions_sha256': sha(args.output / 'predictions.jsonl.gz')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ablation')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if not args.score_only:
        run_inference(args)
    score(args.output)


if __name__ == '__main__':
    main()
