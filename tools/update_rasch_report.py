"""Summarize six paired Rasch comparisons and the model-retention rationale."""
import json
import statistics
from prepare_proxy import ROOT, sha


def main():
    initial = json.loads((ROOT / 'results/ablation/summary.json').read_text())['variants']
    extra = json.loads((ROOT / 'results/rasch_confirmation/summary.json').read_text())['variants']
    with_rasch = initial['full']['per_seed'] + extra['legacy_rasch']['per_seed']
    without = initial['no_rasch']['per_seed'] + extra['without_rasch']['per_seed']
    if any([r['seed'] for r in rows] != list(range(42, 48)) for rows in (with_rasch, without)):
        raise ValueError('Unexpected paired seed coverage')
    metrics = {}
    for name, rows in [('with_rasch', with_rasch), ('without_rasch', without)]:
        metrics[name] = {'brier': [statistics.mean(r['brier'][j] for r in rows) for j in range(6)],
                         'alc': statistics.mean(r['alc'] for r in rows)}
    full, removed = metrics['with_rasch'], metrics['without_rasch']
    zero_wins = sum(a['brier'][0] < b['brier'][0] for a, b in zip(with_rasch, without))
    alc_wins = sum(a['alc'] < b['alc'] for a, b in zip(with_rasch, without))
    if full['brier'][-2:] != removed['brier'][-2:]:
        raise ValueError('Rasch must have no effect at kernel-only checkpoints')
    decision = {
        'action': 'retain_rasch',
        'reason': 'Retain the original low-budget Rasch term: better zero-label Brier in all six paired draws, essentially tied overall local ALC, and a worse subsequent no-Rasch official batch. The official comparison is unpaired and does not establish a causal or universal Rasch advantage.',
        'runtime_reference_commit': 'f9afbd258cc1f47ef158487261c79db14a48eafa',
        'inference_manifest_sha256': sha(ROOT / 'artifacts/manifest.sha256.json'),
        'seeds': list(range(42, 48)), 'metrics': metrics,
        'with_rasch_zero_label_wins': zero_wins, 'with_rasch_alc_wins': alc_wins,
        'scope': 'Same 92 selected public source groups across six draws, matched acquisition/evaluation items, no refitting; seeds are not independent source-held-out tests.',
        'official_evidence': 'results/official/comparison.json',
        'official_evidence_limit': 'Different random draws and dates; only total ALC supplied for the no-Rasch batch; historical submitted ZIP hashes unavailable.',
    }
    (ROOT / 'results/rasch_confirmation/decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    table = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'Paired variant & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\', r'\midrule']
    for label, values in [('With Rasch', full), ('Without Rasch', removed)]:
        table.append(label + ' & ' + ' & '.join(f'{v:.5f}' for v in values['brier'] + [values['alc']]) + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']
    (ROOT / 'report/rasch_retention_table.tex').write_text('\n'.join(table) + '\n')
    text = (r'\paragraph{Why retain Rasch?} '
            + f"Across six matched draws, retaining Rasch lowers zero-label Brier from ${removed['brier'][0]:.5f}$ "
            + f"to ${full['brier'][0]:.5f}$, with the same direction in all {zero_wins} seeds (Table~\\ref{{tab:rasch_retention}}). "
            + f"Overall ALC is essentially tied: ${full['alc']:.5f}$ with Rasch versus ${removed['alc']:.5f}$ without it; "
            + f"retention improves ALC in only {alc_wins} of six seeds. "
            + r'Budgets 15 and 31 are identical because the hybrid has zero weight. '
            + r'\textbf{The controlled evidence supports a small initialization benefit, not an aggregate ALC advantage.} '
            + r'We retain the original limited-weight Rasch term to preserve this cold-start contribution; '
            + r'the less favorable subsequent no-Rasch official batch adds a practical reason to withdraw the simplification (Section~\ref{sec:official}). '
            + r'All six draws reuse the same sources, and the official batches are unpaired, so neither establishes a general transfer benefit.')
    (ROOT / 'report/rasch_retention_analysis.tex').write_text(text + '\n')
    print('Generated six-seed evidence and retention decision without claiming an overall local ALC improvement.')


if __name__ == '__main__':
    main()
