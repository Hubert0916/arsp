"""Paired random/adaptive acquisition on complete groups at three fixed seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import gzip
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from zipfile import ZipFile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

from prepare_proxy import ROOT, sha, write_json, split_group
from smoke_test import load_module


def finite_probability(p):
    p = float(p)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(f'Invalid prediction: {p}')
    return p


def random_choose(input, context):
    """Uniform sampling without replacement, using only current context/input."""
    remaining, total = context['labels_remaining'], context['items_remaining']
    if remaining <= 0:
        return False
    if total <= remaining:
        return True
    raw = json.dumps([context, input], sort_keys=True, separators=(',', ':'))
    u = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], 'big') / 2**64
    return bool(u < remaining / total)


def init_worker(bundle, config):
    global MODEL, LABELING, BASELINE, CONFIG, ORIGINAL_EMBEDDING, VECTOR, VALIDATED
    sys.path.insert(0, str(bundle))
    MODEL = load_module(Path(bundle) / 'model.py', 'asrp_submission')
    LABELING = load_module(Path(bundle) / 'labeling.py', 'asrp_labeling')
    BASELINE = load_module(ROOT / 'baselines/empirical_mean/model.py', 'empirical_baseline')
    CONFIG = config
    VALIDATED = False
    ORIGINAL_EMBEDDING = MODEL._item_embedding
    @lru_cache(maxsize=4096)
    def vector(encoder, content):
        value = ORIGINAL_EMBEDDING(encoder, {'item_content': content})
        value.flags.writeable = False
        return value
    VECTOR = vector
    MODEL._item_embedding = lambda encoder, item: vector(encoder, item.get('item_content', ''))


def component_predictions(input, evidence):
    smooth, kernel, vector, n = MODEL._prediction_components(input, evidence)
    full = MODEL._combine(input, smooth, kernel, vector, n)
    no_kernel = MODEL._calibration_adjust(smooth, input[1], vector, n,
                                         MODEL._resources()['topic_calibration'])
    weight = MODEL._kernel_weight(n)
    no_topic = weight * kernel + (1. - weight) * smooth
    return {k: finite_probability(v) for k,v in
            [('full',full),('no_kernel',no_kernel),('no_topic',no_topic)]}


def validate_components(input, evidence, values):
    config = MODEL._resources()['config']
    weights, adjust = config['probe_weights'], MODEL._calibration_adjust
    try:
        config['probe_weights'] = [0.] * 6
        if abs(MODEL.predict(input, evidence)-values['no_kernel']) > 1e-12:
            raise ValueError('Kernel-removal prediction differs from native inference')
        config['probe_weights'] = weights
        MODEL._calibration_adjust = lambda p, *args: p
        if abs(MODEL.predict(input, evidence)-values['no_topic']) > 1e-12:
            raise ValueError('Calibration-removal prediction differs from native inference')
    finally:
        config['probe_weights'], MODEL._calibration_adjust = weights, adjust


def evaluate_group(group):
    """Each worker keeps only pure feature caches across seeds; no fitted evidence."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        return evaluate_group_inner(group)


def evaluate_group_inner(group):
    global VALIDATED
    start = time.monotonic()
    MODEL.get_predictor().vectors.clear()
    MODEL.get_predictor().fits.clear()
    VECTOR.cache_clear()
    predictions, acquisitions, components = [], [], []
    for seed in CONFIG['seeds']:
        pool, targets = split_group(group, seed)
        warmup = min(16, max(0, len(pool)-31))
        for policy in CONFIG['policies']:
            LABELING._policy = None
            MODEL.get_predictor().fits.clear()
            support, cursor, decisions, ids = [], 0, [], []
            rows = [{'source': group['source'], 'subject_id': group['subject_id'],
                     'seed': seed, 'policy': policy, 'item_id': r['item_id'], 'response_id': r['response_id'],
                     'y': r['label'], 'asrp': [], 'empirical_mean': []} for r in targets]
            component_rows = [{k: v for k,v in row.items() if k not in {'asrp','empirical_mean'}} |
                              {'full': [], 'no_kernel': [], 'no_topic': []} for row in rows]
            for budget in CONFIG['budgets']:
                while len(support) < budget:
                    if cursor >= len(pool):
                        raise ValueError('Acquisition stream exhausted')
                    candidate = pool[cursor]
                    context = {'subject_id': group['subject_alias'] + f'_seed_{seed}',
                               'benchmark_id': group['benchmark_alias'], 'labels_acquired': len(support),
                               'labels_remaining': 31-len(support), 'max_labels': 31,
                               'items_remaining': len(pool)-cursor}
                    reason = ('forced' if context['items_remaining'] <= context['labels_remaining']
                              else 'warmup' if policy == 'adaptive' and cursor < warmup
                              else 'adaptive' if policy == 'adaptive' else 'random')
                    chosen = (random_choose(candidate['input'], context) if policy == 'random'
                              else LABELING.acquisition_function(candidate['input'], labeled=support.copy(), context=context.copy()))
                    if not isinstance(chosen, bool):
                        raise ValueError('Acquisition hook must return bool')
                    decisions.append({'position': cursor, 'item_id': candidate['item_id'],
                                      'acquired': chosen, 'reason': reason})
                    if chosen:
                        support.append([candidate['input'], candidate['label']])
                        ids.append(candidate['item_id'])
                    cursor += 1
                for index, (row, target) in enumerate(zip(rows, targets)):
                    values = component_predictions(target['input'], support.copy())
                    row['asrp'].append(values['full'])
                    for variant, value in values.items():
                        component_rows[index][variant].append(value)
                    if index == 0 and not VALIDATED:
                        validate_components(target['input'], support.copy(), values)
                    row['empirical_mean'].append(finite_probability(BASELINE.predict(target['input'], support.copy())))
                # Verify every checkpoint/trajectory on the first group in each worker.
                if not VALIDATED:
                    cached_vector = MODEL._item_embedding
                    MODEL._item_embedding = ORIGINAL_EMBEDDING
                    try:
                        direct = finite_probability(MODEL.predict(targets[0]['input'], support.copy()))
                    finally:
                        MODEL._item_embedding = cached_vector
                    if abs(direct-rows[0]['asrp'][-1]) > 1e-12:
                        raise ValueError('Feature memoization changed a packaged prediction')
            predictions.extend(rows)
            if policy == 'adaptive':
                components.extend(component_rows)
            acquisitions.append({'source': group['source'], 'subject_id': group['subject_id'],
                                 'seed': seed, 'policy': policy, 'n_items': len(group['items']),
                                 'pool_size': len(pool), 'evaluation_size': len(targets),
                                 'warmup': warmup if policy == 'adaptive' else 0,
                                 'acquired_item_ids': ids, 'labels': [r[1] for r in support], 'decisions': decisions})
    VALIDATED = True
    return group['source'], predictions, acquisitions, components, time.monotonic()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/proxy.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduced')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-groups', type=int, help='Development subset only')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'experiments/proxy_manifest.json').read_text())
    if sha(args.data) != manifest['prepared_sha256']:
        raise ValueError('Prepared data differs from published manifest')
    with args.data.open() as stream:
        groups = [json.loads(line) for line in stream]
    if args.max_groups is not None:
        groups = groups[:args.max_groups]
    args.output.mkdir(parents=True, exist_ok=True)
    chunks = args.output / 'chunks'
    chunks.mkdir(exist_ok=True)
    config = manifest['config']
    with tempfile.TemporaryDirectory(prefix='asrp-evaluate-') as directory:
        root = Path(directory)
        with ZipFile(ROOT / 'dist/asrp.zip') as archive:
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError('Unsafe ZIP path')
            archive.extractall(root)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker, initargs=(root, config)) as executor:
            futures = [executor.submit(evaluate_group, group) for group in groups]
            for number, future in enumerate(as_completed(futures), 1):
                source, predictions, acquisitions, components, seconds = future.result()
                for name, records in [('predictions', predictions), ('acquisition', acquisitions), ('components', components)]:
                    (chunks / f'{source}.{name}.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in records))
                print(f'Evaluated {number}/{len(groups)}: {source} ({seconds:.1f}s; {len(predictions)} prediction rows)', flush=True)
    # Stable ordering is independent of worker scheduling.
    for name in ['predictions', 'acquisition']:
        with (args.output / f'{name}.jsonl').open('w') as stream:
            for group in groups:
                stream.write((chunks / f"{group['source']}.{name}.jsonl").read_text())
    with (args.output / 'component_predictions.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8') as stream:
                for group in groups:
                    stream.write((chunks / f"{group['source']}.components.jsonl").read_text())
    provenance = {'config': config, 'n_groups': len(groups), 'data_sha256': sha(args.data),
                  'split_manifest_sha256': sha(ROOT / 'experiments/proxy_manifest.json'),
                  'inference_manifest_sha256': sha(ROOT / 'artifacts/manifest.sha256.json'),
                  'lockfile_sha256': sha(ROOT / 'uv.lock'), 'python': platform.python_version(),
                  'files': {name: sha(args.output / name) for name in ['predictions.jsonl', 'acquisition.jsonl', 'component_predictions.jsonl.gz']},
                  'scripts': {str(p.relative_to(ROOT)): sha(p) for p in
                              [ROOT / 'tools/prepare_proxy.py', Path(__file__), ROOT / 'baselines/empirical_mean/model.py']}}
    write_json(args.output / 'run.json', provenance)


if __name__ == '__main__':
    main()
