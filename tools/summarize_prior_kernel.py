"""Write a concise, reproducible account of the locked no-Rasch experiment."""
import gzip
import json

from prepare_proxy import ROOT, sha, write_json
from search_prior_kernel import CANDIDATE, ALC, curve, load_cache


def main():
    folder = ROOT / 'results/asrp_next'
    manifest = ROOT / 'experiments/asrp_next/validation_manifest.json'
    if manifest.exists():
        manifest.with_suffix('.json.gz').write_bytes(gzip.compress(manifest.read_bytes(), mtime=0))
    for phase in ['development', 'validation']:
        trace = folder / f'{phase}_cache/acquisition.json'
        if trace.exists():
            trace.with_suffix('.json.gz').write_bytes(gzip.compress(trace.read_bytes(), mtime=0))
    development = json.loads((folder / 'development/search.json').read_text())
    validation = json.loads((folder / 'validation/summary.json').read_text())
    config = validation['config']
    display = {'full': 'Released ASRP (Rasch)', 'no_rasch': 'Plain no-Rasch',
               'candidate': 'Refined no-Rasch'}
    cache, _ = load_cache(folder / 'development_cache')
    names = ['candidate', 'full', 'no_rasch', 'em'] + CANDIDATE.ablations(config)
    development_rows = []
    for record, parts in cache:
        methods = {}
        for name in names:
            prediction = parts[name] if name in ['full', 'no_rasch', 'em'] else CANDIDATE.probabilities(
                parts, config, None if name == 'candidate' else name)
            brier = curve(parts, prediction)
            methods[name] = {'brier': brier.tolist(), 'alc': float(brier @ ALC)}
        development_rows.append({'draw_id': record['draw_id'], 'policy': record['policy'], 'methods': methods})
    write_json(folder / 'development/selected_draws.json', development_rows)
    lines = [
        '# No-Rasch candidate evaluation', '',
        '**Decision: ' + ('passes the predefined local validation gates.' if validation['passed']
                         else 'does not pass the predefined local validation gates; retain released ASRP.') + '**', '',
        f"One global configuration was selected from {development['n_configurations']} development settings, "
        'using 30 previously released draws and both acquisition policies. It was then locked before scoring '
        '40 broad validation draws and 20 seven-benchmark-pool stress draws. The validation draws contain '
        '176 distinct groups excluded from the previous proxy and formative development groups.', '',
        'This is a public approximation of the documented formative protocol. The private coordinator '
        'and private pool are unavailable. Frozen prior/calibration training still overlaps the public '
        'release; the new groups are held out from candidate selection, not from all offline fitting.', '',
        '## Selected configuration', '', '```json', json.dumps(config, indent=2), '```', '',
        'For the selected fixed-strength configuration, the prediction is '
        '`clip((s + 4*q)/(n + 4) + w_n*r + 8/(8+n)*delta, 0, 1)`. '
        '`q` is the frozen metadata prior, `r` is the support-conditioned kernel residual, '
        'and `delta` is the frozen topic correction. The kernel weights remain '
        '`[0, 0, 0.2, 0.4, 1, 1]`. The candidate retains the metadata-smoothed estimate '
        'at high budgets instead of replacing it with the original kernel intercept. '
        'Evidence-dependent concentration mixing and extra prior shrinkage were explored '
        'but are not used by this selected configuration.', '',
        '## Matched validation ALC', '',
        '| Pool | Acquisition | Released ASRP | Plain no-Rasch | Candidate | Empirical Mean |',
        '|---|---|---:|---:|---:|---:|',
    ]
    assert config == {'prior_shrink': 1., 'prior_center': .5, 'concentrations': [4.],
                      'masses': [1.], 'residual_scale': 1., 'combine': 'rebase', 'topic_decay': 8.}, \
        'Update the selected-formula description for a changed configuration.'
    for cohort, policies in validation['cohorts'].items():
        for policy, value in policies.items():
            methods = value['methods']
            scores = ' | '.join(f"{methods[m]['alc']:.6f}" for m in ['full', 'no_rasch', 'candidate', 'em'])
            lines.append(f'| {cohort} | {policy} | {scores} |')
    lines += ['', 'All methods receive exactly the same acquired labels within each draw and acquisition '
              'policy. Lower is better. The two pools are reported separately.', '',
              '## Adaptive: six budgets', '',
              '| Pool | Method | 0 | 1 | 3 | 7 | 15 | 31 | ALC |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort, policies in validation['cohorts'].items():
        methods = policies['adaptive']['methods']
        for method in ['full', 'no_rasch', 'candidate']:
            scores = ' | '.join(f'{v:.6f}' for v in methods[method]['brier'] + [methods[method]['alc']])
            lines.append(f'| {cohort} | {display[method]} | {scores} |')
    lines += ['', '## Adaptive: paired differences and ablations', '',
              '| Pool | Comparison | Mean ALC difference | Paired bootstrap 95% interval |',
              '|---|---|---:|---|']
    for cohort, policies in validation['cohorts'].items():
        value = policies['adaptive']
        comparisons = [('Candidate minus released ASRP', value['candidate_minus_reference']['full'])]
        comparisons += [(f'{name} minus candidate', r) for name, r in value['ablation_minus_candidate'].items()]
        for title, r in comparisons:
            lo, hi = r['percentile_95_interval']
            lines.append(f"| {cohort} | {title} | {r['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '`no_kernel` removes the kernel residual; `no_topic` removes the complete topic correction; '
              '`neutral_metadata_prior` replaces the metadata prior with 0.5; '
              '`original_kernel_blend` restores the previous kernel-intercept blend while retaining '
              'candidate topic decay; `no_topic_decay` keeps topic correction at its original strength.', '',
              'A negative candidate-minus-reference value favors the candidate. A positive '
              'ablation-minus-candidate value favors retaining the component. Intervals resample '
              'draws within the fixed evaluation pool; draws can reuse groups. These intervals '
              'do not establish generalization to a new private pool.', '', '## Acceptance gates', '']
    for gate in validation['gates']:
        lines.append(f"- {'PASS' if gate['passed'] else 'FAIL'}: {gate['name']}")
    lines += ['', '## Reproduction', '',
              'See [protocol and commands](../../experiments/asrp_next/README.md). '
              '`development/search.json` contains all development settings; `validation/draws.json` '
              'contains every scored validation draw. `validation/summary.json` contains the '
              'locked configuration, hashes, comparisons, and gate decisions.', '',
              f"Original-curve reconstruction maximum Brier error: {development['cache_max_reference_brier_error']:.3g}. "
              f"Plain no-Rasch reconstruction maximum probability error: {development['plain_no_rasch_max_probability_error']:.3g}.", '']
    runtime_path = folder / 'runtime_check.json'
    phases = [folder / f'{phase}_runtime_check.json' for phase in ['development', 'validation']]
    if not runtime_path.exists() and all(p.exists() for p in phases):
        left, right = [json.loads(p.read_text()) for p in phases]
        assert left['locked_candidate_sha256'] == right['locked_candidate_sha256'] == validation['locked_candidate_sha256']
        assert left['native_files'] == right['native_files']
        combined = {
            'locked_candidate_sha256': left['locked_candidate_sha256'],
            'predictions_checked': left['predictions_checked'] + right['predictions_checked'],
            'full_acquisition_trajectories_checked': left['full_acquisition_trajectories_checked'] + right['full_acquisition_trajectories_checked'],
            'max_probability_error': max(left['max_probability_error'], right['max_probability_error']),
            'draws': left['draws'] + right['draws'], 'native_files': left['native_files'],
            'phase_check_sha256': {p.name: sha(p) for p in phases},
            'scope': 'First and last evaluation inputs of every group in the listed draws, all six budgets, both policies.',
        }
        write_json(runtime_path, combined)
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        assert runtime['locked_candidate_sha256'] == validation['locked_candidate_sha256']
        lines += [f"Standalone runtime: {runtime['predictions_checked']} prediction checks, "
                  f"{runtime['full_acquisition_trajectories_checked']} complete acquisition replays, "
                  f"maximum probability error {runtime['max_probability_error']:.3g}. "
                  'The standalone runtime contains no Rasch difficulty weights or subject-ability model.', '']
    if (folder / 'submission.json').exists():
        packaged = json.loads((folder / 'submission.json').read_text())
        assert packaged['locked_candidate_sha256'] == validation['locked_candidate_sha256']
        lines += ['An experimental ZIP can be built with `tools/package_prior_kernel.py`; '
                  '`submission.json` records its checksum. It does not replace the released submission. '
                  '`interface_checks.json` records the ZIP smoke test and the public starting-kit interface check; '
                  'these checks are not competition performance estimates.', '']
    (folder / 'README.md').write_text('\n'.join(lines))
    print(folder / 'README.md')


if __name__ == '__main__':
    main()
