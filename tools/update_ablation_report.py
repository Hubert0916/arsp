"""Audit and render component removals for the current prior--kernel predictor."""
import argparse
import json
from pathlib import Path
from score_ablation import ROOT, VARIANTS, score

LABELS={'full':'Full ASRP','no_kernel':'No Kernel Probe','no_topic':'No topic calibration'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=ROOT/'results/proxy')
    args=parser.parse_args()
    result=score(args.directory)
    lines=[r'\begin{tabular}{lrrrrrrrr}',r'\toprule',
           r'Variant & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC & $\Delta$ALC\\',r'\midrule']
    for key in VARIANTS:
        v=result['variants'][key]
        delta='--' if key=='full' else f"${v['delta_alc_vs_full']:+.4f}$"
        lines.append(LABELS[key]+' & '+' & '.join(f'{x:.4f}' for x in v['brier']+[v['alc']])+' & '+delta+r'\\')
    (ROOT/'report/ablation_table.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    full,kernel,topic=(result['variants'][k] for k in VARIANTS)
    empirical=json.loads((args.directory/'summary.json').read_text())['combinations']['empirical_mean_adaptive']['alc']
    def description(value):
        delta=value['delta_alc_vs_full']
        return f"{'increases' if delta>0 else 'decreases'} ALC by ${abs(delta):.4f}$"
    text=(r'\paragraph{Local adaptation and calibration.} '
          +f"Removing Kernel Probe {description(kernel)}, from ${full['alc']:.4f}$ to ${kernel['alc']:.4f}$. "
          +f"At budget 31, Brier changes from ${full['brier'][-1]:.4f}$ to ${kernel['brier'][-1]:.4f}$. "
          +r'Budgets zero and one are unchanged because the kernel weight is already zero. '
          +(f"The predictor without Kernel Probe still improves on matched-label Empirical Mean (${empirical:.4f}$), "
            +r'showing that the retained prior smoothing and calibration preserve a substantial part of the advantage. '
            if kernel['alc'] < empirical else '')
          +f"Removing topic calibration {description(topic)}, to ${topic['alc']:.4f}$. "
          +r'These comparisons isolate removals from the current frozen configuration, with metadata smoothing retained and identical acquired evidence.')
    (ROOT/'report/ablation_analysis.tex').write_text(text+'\n')
    print('Updated current component table and analysis.')


if __name__=='__main__': main()
