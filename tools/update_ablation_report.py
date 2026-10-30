"""Audit released ablation predictions and generate the six-budget report table."""
import argparse
from pathlib import Path

from evaluate_ablation import ROOT, VARIANTS, score

LABELS = {'full': 'Full ASRP', 'no_rasch': 'No Rasch',
          'no_kernel': 'No Kernel Probe', 'no_topic': 'No topic calibration'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=ROOT / 'results/ablation')
    args = parser.parse_args()
    result = score(args.directory)
    lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
             r'Variant & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC & $\Delta$ALC\\',
             r'\midrule']
    for key in VARIANTS:
        value = result['variants'][key]
        metrics = ' & '.join(f'{x:.4f}' for x in value['brier'] + [value['alc']])
        delta = '--' if key == 'full' else f"${value['delta_alc_vs_full']:+.4f}$"
        lines.append(LABELS[key] + ' & ' + metrics + ' & ' + delta + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT / 'report/ablation_table.tex').write_text('\n'.join(lines) + '\n')
    values = result['variants']
    full, rasch, kernel, topic = (values[k] for k in VARIANTS)
    # These conclusions describe the frozen published run; fail instead of silently
    # recycling the interpretation if a future experiment reverses the effects.
    if not (kernel['delta_alc_vs_full'] > topic['delta_alc_vs_full'] > 0
            and rasch['delta_alc_vs_full'] < 0 and rasch['brier'][0] > full['brier'][0]):
        raise ValueError('Component effects changed; revise the interpretation before publishing')
    analysis = (
        r'\paragraph{Kernel adaptation contributes at higher budgets.} '
        + f"Removing the kernel branch increases ALC from ${full['alc']:.4f}$ to ${kernel['alc']:.4f}$, "
        + f"the largest increase among the three removals (${kernel['delta_alc_vs_full']:.4f}$). "
        + f"At budget 31, Brier worsens from ${full['brier'][-1]:.4f}$ to ${kernel['brier'][-1]:.4f}$. "
        + r'This supports the kernel branch at higher budgets; its zero weight leaves budgets zero and one unchanged.' + '\n\n'
        + r'\paragraph{Calibration provides a smaller additional gain.} '
        + f"Removing topic calibration increases ALC to ${topic['alc']:.4f}$ "
        + f"(${topic['delta_alc_vs_full']:+.4f}$); its largest Brier benefit is at budget zero. "
        + r'The Rasch effect is examined by budget and prior availability below; '
        + r'the full predictor advantage over Empirical Mean cannot be assigned to a single component.'
    )
    (ROOT / 'report/ablation_analysis.tex').write_text(analysis + '\n')
    print('Generated report/ablation_table.tex from audited component predictions.')


if __name__ == '__main__':
    main()
