"""Render current-version official results or an explicit pending placeholder."""
import hashlib
import json

from parse_official_results import ROOT, summarize


def main():
    directory = ROOT / 'results/official'
    report = ROOT / 'report'
    state = json.loads((directory / 'status.json').read_text())
    version = json.loads((ROOT / 'artifacts/asrp/config.json').read_text())['version']
    digest = hashlib.sha256((ROOT / 'artifacts/manifest.sha256.json').read_bytes()).hexdigest()
    if state['model_version'] != version or state['inference_manifest_sha256'] != digest:
        raise ValueError('Official-evaluation status belongs to different inference files')

    table = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'ASRP + Adaptive & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\',
             r'\midrule']
    if state['status'] == 'pending':
        if state['n_evaluations'] != 0:
            raise ValueError('Pending status must have zero completed evaluations')
        overview = (r'\textbf{Pending.} Official six-budget Brier scores and Brier ALC for the '
                    r'current ASRP version will be added after Codabench evaluation.')
        table.append('Pending & ' + ' & '.join(['--'] * 7) + r'\\')
    elif state['status'] == 'complete':
        provenance = json.loads((directory / 'provenance.json').read_text())
        if (provenance.get('inference_manifest_sha256') != digest
                or provenance.get('model_version') != version
                or provenance.get('acquisition_policy_confirmed_by_submitter') != state['acquisition_policy']
                or not provenance.get('same_submission_zip_confirmed_by_submitter')):
            raise ValueError('Feedback provenance must identify the current model and the same evaluated ZIP')
        runs = json.loads((directory / 'runs.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        if summarize(runs) != summary or summary['n_complete_blocks'] != state['n_evaluations']:
            raise ValueError('Official aggregates disagree with their recorded evaluation count')
        submissions = json.loads((directory / 'submissions.json').read_text())['submissions']
        by_block = {s['feedback_block']: s for s in submissions}
        if set(by_block) != {r['block'] for r in runs} or len({s['submission_id'] for s in submissions}) != len(runs):
            raise ValueError('Submission IDs must map one-to-one to feedback records')
        for run in runs:
            if abs(run['brier_alc'] - by_block[run['block']]['displayed_brier_alc']) > 1.00001e-6:
                raise ValueError('Submission score disagrees with the reconstructed ALC')
        metrics = summary['complete_six_budget']
        for key, label in [('mean', 'Mean'), ('sample_sd', 'SD')]:
            table.append(label + ' & ' + ' & '.join(
                f'{m[key]:.4f}' for m in metrics['brier'] + [metrics['brier_alc']]) + r'\\')
        overview = (f"Official results summarize {state['n_evaluations']} evaluations of the same "
                    r'current ASRP submission with adaptive acquisition, using equal group weighting '
                    r'within each evaluation and equal weighting across evaluations.')
    else:
        raise ValueError('Unknown official-evaluation status')
    table.extend([r'\bottomrule', r'\end{tabular}'])
    (report / 'official_overview.tex').write_text(overview + '\n')
    (report / 'official_table.tex').write_text('\n'.join(table) + '\n')
    print(f"Updated official report: {state['status']} ({version}).")


if __name__ == '__main__':
    main()
