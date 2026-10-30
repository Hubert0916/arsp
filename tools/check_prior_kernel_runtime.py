"""Check the standalone no-Rasch runtime against cached component predictions."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from cache_prior_kernel import compact
from draw_protocol import BUDGETS, fixed_split, input_pair, random_choose, trajectory
from prepare_proxy import ROOT, sha, write_json
from search_prior_kernel import CANDIDATE
from smoke_test import load_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'data/asrp_next_bundle')
    parser.add_argument('--phase', choices=['all', 'development', 'validation'], default='all')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    assert not (bundle / 'difficulty_model.npz').exists()
    assert not (bundle / 'subject_prior.json').exists()
    sys.path.insert(0, str(bundle))
    model = load_module(bundle / 'model.py', 'native_candidate')
    labeling = load_module(bundle / 'labeling.py', 'candidate_labeling')
    lock = json.loads((ROOT / 'experiments/asrp_next/locked_candidate.json').read_text())
    assert model._resources()['config'] == lock['config']
    maximum = 0.
    checked = 0
    trace_checks = 0
    evaluated = []
    for phase, manifest_path, data_path in [
        ('development', 'experiments/formative_manifest.json', 'data/formative.jsonl'),
        ('validation', 'experiments/asrp_next/validation_manifest.json', 'data/asrp_next_validation.jsonl'),
    ]:
        if args.phase != 'all' and phase != args.phase:
            continue
        manifest = json.loads((ROOT / manifest_path).read_text())
        draws = [manifest['draws'][0]]
        if phase == 'validation':
            draws.append(next(d for d in manifest['draws'] if d['cohort'] == 'small_pool'))
        wanted = {g for d in draws for g in d['group_ids']}
        groups = {}
        with (ROOT / data_path).open() as f:
            for line in f:
                g = json.loads(line)
                if g['group_id'] in wanted:
                    groups[g['group_id']] = compact(g, manifest['config']['snapshot_seed'])
        cache_dir = ROOT / f'results/asrp_next/{phase}_cache'
        trace_path = cache_dir / 'acquisition.json'
        if trace_path.exists():
            saved_traces = json.loads(trace_path.read_text())
        elif trace_path.with_suffix('.json.gz').exists():
            saved_traces = json.loads(gzip.decompress(trace_path.with_suffix('.json.gz').read_bytes()))
        else:
            saved_traces = [r for d in draws for r in json.loads(
                (cache_dir / 'trace_chunks' / f"{d['draw_id']:03d}.json").read_text())]
        trace = {(r['draw_id'], r['policy']): r for r in saved_traces}
        for draw in draws:
            gg = [groups[g] for g in draw['group_ids']]
            targets, selected = [], []
            for g in gg:
                start = len(targets)
                _, evaluation = fixed_split(g, manifest['config']['snapshot_seed'])
                for item in evaluation:
                    targets.extend(input_pair(g, item, r) for r in item['responses'])
                selected.extend([start, len(targets) - 1])
            for policy in ['random', 'adaptive']:
                labeling._policy = None
                model.get_predictor().fits.clear()
                parts = dict(np.load(cache_dir / f"{draw['draw_id']:03d}_{policy}.npz"))
                expected = CANDIDATE.probabilities(parts, lock['config'])
                assert len(targets) == len(expected)
                def choose(x, evidence, context):
                    return random_choose(x, context) if policy == 'random' else labeling.acquisition_function(x, evidence, context)
                def checkpoint(budget, evidence, counts):
                    nonlocal maximum, checked
                    assert all(n == budget for n in counts)
                    for row in selected:
                        prediction = model.predict(targets[row], evidence)
                        error = abs(prediction - expected[row, BUDGETS.index(budget)])
                        maximum = max(maximum, error)
                        checked += 1
                        assert error < 1e-10, (phase, draw['draw_id'], policy, budget, row, error)
                decisions, counts = trajectory(gg, manifest['config']['snapshot_seed'], choose, checkpoint)
                assert decisions == trace[(draw['draw_id'], policy)]['decisions']
                assert counts == trace[(draw['draw_id'], policy)]['counts']
                trace_checks += 1
                evaluated.append({'phase': phase, 'draw_id': draw['draw_id'], 'policy': policy, 'groups': draw['group_ids']})
                print(f'Native runtime verified: {phase}, draw {draw["draw_id"]}, {policy}', flush=True)
    result = {
        'predictions_checked': checked, 'max_probability_error': maximum,
        'full_acquisition_trajectories_checked': trace_checks, 'draws': evaluated,
        'locked_candidate_sha256': sha(ROOT / 'experiments/asrp_next/locked_candidate.json'),
        'native_files': {p.relative_to(bundle).as_posix(): sha(p) for p in sorted(bundle.rglob('*'))
                         if p.is_file() and '__pycache__' not in p.parts},
        'scope': f'Phase={args.phase}. First and last evaluation inputs of every group in the listed draws, all six budgets, both acquisition policies.',
    }
    filename = 'runtime_check.json' if args.phase == 'all' else f'{args.phase}_runtime_check.json'
    write_json(ROOT / 'results/asrp_next' / filename, result)
    print(f'Passed {checked} probability checks and {trace_checks} acquisition replays; max error={maximum:.3g}')


if __name__ == '__main__':
    main()
