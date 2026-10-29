"""Generate the paired Rasch retention/removal table from both draw sets."""
import json
import statistics
from pathlib import Path
from prepare_proxy import ROOT


def main():
    earlier=json.loads((ROOT/'results/legacy_rasch/summary.json').read_text())
    check=json.loads((ROOT/'results/rasch_confirmation/summary.json').read_text())
    initial_old=earlier['variants']['full'];initial_new=earlier['variants']['no_rasch']
    extra_old=check['variants']['legacy_rasch'];extra_new=check['variants']['without_rasch']
    old_seeds=initial_old['per_seed']+extra_old['per_seed']
    new_seeds=initial_new['per_seed']+extra_new['per_seed']
    if [r['seed'] for r in old_seeds]!=[42,43,44,45,46,47] or [r['seed'] for r in new_seeds]!=[42,43,44,45,46,47]:
        raise ValueError('Unexpected Rasch comparison seed coverage')
    old_mean=statistics.mean(r['alc'] for r in old_seeds)
    new_mean=statistics.mean(r['alc'] for r in new_seeds)
    diffs=[n['alc']-o['alc'] for o,n in zip(old_seeds,new_seeds)]
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',
           r'Draw set & With Rasch & Without Rasch & $\Delta$ALC\\',r'\midrule']
    for label,a,b in [('Initial seeds 42--44',initial_old['alc'],initial_new['alc']),
                      ('Additional seeds 45--47',extra_old['alc'],extra_new['alc']),
                      ('All six seeds',old_mean,new_mean)]:
        lines.append(f'{label} & {a:.5f} & {b:.5f} & ${b-a:+.5f}$'+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'report/rasch_removal_table.tex').write_text('\n'.join(lines)+'\n')
    old_zero=statistics.mean(r['brier'][0] for r in old_seeds)
    new_zero=statistics.mean(r['brier'][0] for r in new_seeds)
    direction='lower' if new_mean<old_mean else 'higher'
    text=(f"Across the six paired draws, removing Rasch produces {direction} mean ALC by ${abs(new_mean-old_mean):.5f}$ "
          +f"and lower ALC in {sum(d<0 for d in diffs)} of the six seeds (Table~\\ref{{tab:rasch_removal}}). "
          +f"Mean zero-label Brier changes from ${old_zero:.4f}$ to ${new_zero:.4f}$. "
          +r'The additional seeds use the same selected sources and frozen artifacts; they are a check of draw sensitivity, not a new-source generalization test. ')
    decision=json.loads((ROOT/'results/rasch_confirmation/decision.json').read_text())
    if decision['action']!='remove_rasch': raise ValueError('Report architecture differs from the recorded decision')
    text+=r'We adopt the simpler prior--kernel predictor, eliminating difficulty inference and ability updates. The measured change is small and includes a cold-start tradeoff; we do not claim uniform improvement across sources.'
    (ROOT/'report/rasch_removal_analysis.tex').write_text(text+'\n')
    print('Updated Rasch removal comparison from six paired draws.')


if __name__=='__main__': main()
