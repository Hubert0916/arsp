"""Resume identical component inference, retaining completed draw caches.

Only worker scheduling changes. Completed arrays are kept byte-for-byte;
their acquisition trajectories are regenerated if the interrupted coordinator
had not yet saved them. No candidate scores are computed here.
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

import cache_prior_kernel as original
from draw_protocol import trajectory, random_choose
from prepare_proxy import ROOT, sha, write_json


def replay_only(task):
    from threadpoolctl import threadpool_limits
    draw, groups = task
    runtime = original.runtime
    results = []
    for policy in runtime.CONFIG['policies']:
        runtime.LABELING._policy = None
        def choose(x, evidence, context):
            return random_choose(x, context) if policy == 'random' else runtime.LABELING.acquisition_function(x, labeled=evidence, context=context)
        def checkpoint(budget, evidence, counts):
            assert all(n == budget for n in counts)
        with threadpool_limits(limits=2):
            decisions, counts = trajectory(groups, runtime.CONFIG['snapshot_seed'], choose, checkpoint)
        results.append({'draw_id': draw['draw_id'], 'policy': policy, 'groups': draw['group_ids'],
                        'decisions': decisions, 'counts': counts})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    if (args.output / 'index.json').exists():
        raise SystemExit('Cache already complete; use the original index.')
    manifest = json.loads(args.manifest.read_text())
    config = manifest['config']
    assert sha(args.data) == manifest['prepared_sha256']
    groups = {}
    with args.data.open() as f:
        for line in f:
            g = original.compact(json.loads(line), config['snapshot_seed'])
            groups[g['group_id']] = g
    frozen = json.loads((ROOT / 'artifacts/manifest.sha256.json').read_text())
    complete = []
    remaining = []
    preserved = {}
    for d in manifest['draws']:
        files = [args.output / f"{d['draw_id']:03d}_{policy}.npz" for policy in config['policies']]
        if all(path.exists() for path in files):
            for path in files:
                with np.load(path) as arrays:
                    assert arrays['full'].shape[1] == 6 and np.all(np.isfinite(arrays['full']))
                preserved[path.name] = sha(path)
            complete.append(d)
        else:
            remaining.append(d)
    print(f'Resuming: preserve {len(complete)} draws, compute {len(remaining)} draws.', flush=True)
    traces = []
    with tempfile.TemporaryDirectory(prefix='asrp-components-resume-') as td:
        bundle = Path(td)
        for name, digest in frozen.items():
            assert sha(ROOT / name) == digest
            prefix = 'src/asrp' if name.startswith('src/') else 'artifacts/asrp'
            target = bundle / Path(name).relative_to(prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=original.initialize, initargs=(bundle, config)) as pool:
            futures = {pool.submit(original.run_draw, (d, [groups[g] for g in d['group_ids']])): ('inference', d)
                       for d in remaining}
            futures.update({pool.submit(replay_only, (d, [groups[g] for g in d['group_ids']])): ('replay', d)
                            for d in complete})
            for i, future in enumerate(as_completed(futures), 1):
                kind, d = futures[future]
                if kind == 'inference':
                    records, tt, elapsed = future.result()
                    for row in records:
                        np.savez_compressed(args.output / f"{row['draw_id']:03d}_{row['policy']}.npz", **row['arrays'])
                    print(f'Completed inference for draw {d["draw_id"]} ({elapsed:.1f}s); {i}/{len(futures)} tasks.', flush=True)
                else:
                    tt = future.result()
                    print(f'Replayed acquisition for preserved draw {d["draw_id"]}; {i}/{len(futures)} tasks.', flush=True)
                traces.extend(tt)
                write_json(args.output / 'trace_chunks' / f"{d['draw_id']:03d}.json", tt)
    for name, digest in preserved.items():
        assert sha(args.output / name) == digest
    index = []
    for d in manifest['draws']:
        for policy in config['policies']:
            name = f"{d['draw_id']:03d}_{policy}.npz"
            index.append({'draw_id': d['draw_id'], 'policy': policy, 'groups': d['group_ids'],
                          'file': name, 'sha256': sha(args.output / name)})
    write_json(args.output / 'index.json', {'manifest_sha256': sha(args.manifest), 'data_sha256': sha(args.data),
        'inference_manifest_sha256': sha(ROOT / 'artifacts/manifest.sha256.json'),
        'script_sha256': sha(Path(original.__file__)), 'resume_script_sha256': sha(Path(__file__)),
        'preserved_files': preserved, 'draws': sorted(index, key=lambda r: (r['draw_id'], r['policy']))})
    write_json(args.output / 'acquisition.json', sorted(traces, key=lambda r: (r['draw_id'], r['policy'])))


if __name__ == '__main__':
    main()
