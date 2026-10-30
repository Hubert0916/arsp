"""Compare Rasch retention/removal on three additional paired draws.

The runtime is read from a pinned Git commit so the paired comparison remains
fixed independently of later submission edits.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

from prepare_proxy import ROOT, sha, split_group, write_json
from smoke_test import load_module

VARIANTS = ('legacy_rasch', 'without_rasch')


def initialize(bundle, legacy_evaluator, config):
    global RUNTIME
    RUNTIME = load_module(Path(legacy_evaluator), 'legacy_evaluator')
    RUNTIME.init_worker(bundle, config)
    model = RUNTIME.MODEL
    original = model.predict
    def without_rasch(input, labeled=None):
        settings = model._resources()[4]
        lambdas = settings['lambdas']
        settings['lambdas'] = [0.] * 6
        try:
            return original(input, labeled)
        finally:
            settings['lambdas'] = lambdas
    model.predict = without_rasch
    # Reuse the paired evaluator's two output slots, then rename explicitly.
    RUNTIME.BASELINE.predict = original


def evaluate(group):
    source, rows, trajectories, elapsed = RUNTIME.evaluate_group(group)
    for row in rows:
        row['without_rasch'] = row.pop('asrp')
        row['legacy_rasch'] = row.pop('empirical_mean')
    return source, rows, trajectories, elapsed


def read_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def save_compressed(path, chunks):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8') as stream:
                for chunk in chunks:
                    stream.write(chunk.read_text())


def score(directory):
    protocol = json.loads((ROOT / 'experiments/rasch_confirmation.json').read_text())
    manifest = json.loads((ROOT / 'experiments/proxy_manifest.json').read_text())
    run = json.loads((directory / 'run.json').read_text())
    if run['protocol'] != protocol or run['manifest_sha256'] != sha(ROOT / 'experiments/proxy_manifest.json'):
        raise ValueError('Confirmation protocol or source manifest changed')
    for name, checksum in run['files'].items():
        if sha(directory / name) != checksum:
            raise ValueError('Confirmation output checksum mismatch')
    rows = read_rows(directory / 'predictions.jsonl.gz')
    traces = read_rows(directory / 'acquisition.jsonl.gz')
    expected = {(g['source'], g['subject_id'], seed) for g in manifest['groups'] for seed in protocol['seeds']}
    cells = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['source'], row['subject_id'], row['seed'])
        identity = (*key, row['item_id'])
        if key not in expected or identity in seen or row['policy'] != 'adaptive' or row['y'] not in (0, 1):
            raise ValueError('Unexpected or duplicate confirmation row')
        seen.add(identity)
        for variant in VARIANTS:
            if len(row[variant]) != 6 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row[variant]):
                raise ValueError('Invalid confirmation probabilities')
        if row['legacy_rasch'][-2:] != row['without_rasch'][-2:]:
            raise ValueError('Rasch removal changed a budget with zero Rasch contribution')
        cells[key].append(row)
    if set(cells) != expected or len(traces) != len(expected):
        raise ValueError('Incomplete confirmation coverage')
    traced = set()
    for trace in traces:
        key = (trace['source'], trace['subject_id'], trace['seed'])
        if key not in expected or key in traced:
            raise ValueError('Invalid confirmation acquisition coverage')
        traced.add(key)
        ids = trace['acquired_item_ids']
        if len(ids) != 31 or len(set(ids)) != 31 or len(cells[key]) != trace['evaluation_size']:
            raise ValueError('Incomplete evidence or evaluation pool')
        if set(ids) & {r['item_id'] for r in cells[key]}:
            raise ValueError('Acquisition/evaluation overlap')
    grouped = []
    weights = manifest['config']['alc_weights']
    for (source, subject, seed), members in sorted(cells.items()):
        record = {'source': source, 'subject_id': subject, 'seed': seed, 'n_rows': len(members)}
        for variant in VARIANTS:
            curve = [math.fsum((r[variant][j]-r['y'])**2 for r in members)/len(members) for j in range(6)]
            record[variant] = {'brier': curve, 'alc': math.fsum(x*w for x,w in zip(curve,weights))}
        record['delta_alc'] = record['without_rasch']['alc']-record['legacy_rasch']['alc']
        grouped.append(record)
    variants = {}
    for variant in VARIANTS:
        per_seed = []
        for seed in protocol['seeds']:
            subset = [r[variant] for r in grouped if r['seed'] == seed]
            per_seed.append({'seed': seed, 'alc': statistics.mean(r['alc'] for r in subset),
                             'brier': [statistics.mean(r['brier'][j] for r in subset) for j in range(6)]})
        variants[variant] = {'brier': [statistics.mean(r['brier'][j] for r in per_seed) for j in range(6)],
                             'alc': statistics.mean(r['alc'] for r in per_seed), 'per_seed': per_seed}
    source_deltas = defaultdict(list)
    for row in grouped:
        source_deltas[row['source']].append(row['delta_alc'])
    deltas = [statistics.mean(v) for v in source_deltas.values()]
    summary = {'protocol': protocol, 'n_groups': len(manifest['groups']), 'n_rows': len(rows), 'variants': variants,
               'delta_alc': variants['without_rasch']['alc']-variants['legacy_rasch']['alc'],
               'sources_improved': sum(x < -1e-12 for x in deltas),
               'sources_worsened': sum(x > 1e-12 for x in deltas),
               'sources_tied': sum(abs(x) <= 1e-12 for x in deltas)}
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'by_source.json', grouped)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/rasch_confirmation')
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if args.score_only:
        score(args.output)
        return
    protocol = json.loads((ROOT / 'experiments/rasch_confirmation.json').read_text())
    manifest = json.loads((ROOT / 'experiments/proxy_manifest.json').read_text())
    if sha(ROOT / 'data/proxy.jsonl') != manifest['prepared_sha256']:
        raise ValueError('Prepare the pinned public proxy first')
    config = dict(manifest['config'], seeds=protocol['seeds'], policies=protocol['policies'])
    groups = read_rows(ROOT / 'data/proxy.jsonl')
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='asrp-rasch-check-') as temp:
        root = Path(temp)
        archive_path = root / 'legacy.tar'
        with archive_path.open('wb') as stream:
            subprocess.run(['git','archive',protocol['legacy_commit'],'src/asrp','artifacts/asrp',
                            'artifacts/manifest.sha256.json','tools/evaluate_proxy.py'], cwd=ROOT, stdout=stream, check=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(root / 'legacy', filter='data')
        bundle = root / 'bundle'
        for name in ['src/asrp','artifacts/asrp']:
            shutil.copytree(root / 'legacy' / name, bundle, dirs_exist_ok=True)
        chunks = root / 'chunks'; chunks.mkdir()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(bundle,root/'legacy/tools/evaluate_proxy.py',config)) as executor:
            futures = [executor.submit(evaluate,g) for g in groups]
            for index,future in enumerate(as_completed(futures),1):
                source, rows, traces, elapsed = future.result()
                for name,values in [('predictions',rows),('acquisition',traces)]:
                    (chunks/f'{source}.{name}.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in values))
                print(f'Confirmed {index}/{len(groups)}: {source} ({elapsed:.1f}s)',flush=True)
        for name in ['predictions','acquisition']:
            save_compressed(args.output/f'{name}.jsonl.gz',[chunks/f"{g['source']}.{name}.jsonl" for g in groups])
    write_json(args.output/'run.json',{'protocol':protocol,'manifest_sha256':sha(ROOT/'experiments/proxy_manifest.json'),
               'prepared_data_sha256':manifest['prepared_sha256'],'script_sha256':sha(Path(__file__)),
               'files':{f'{name}.jsonl.gz':sha(args.output/f'{name}.jsonl.gz') for name in ['predictions','acquisition']}})
    score(args.output)


if __name__ == '__main__':
    main()
