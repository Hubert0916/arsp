"""Render the factorial prior study without inferring private-data effects."""
import json
from prepare_proxy import ROOT, sha


def main():
    directory=ROOT/'results/rasch_prior_ablation'
    run=json.loads((directory/'run.json').read_text())
    if sha(directory/'predictions.jsonl.gz')!=run['predictions_sha256']:
        raise ValueError('Factorial prediction checksum mismatch')
    for name,checksum in run['inputs'].items():
        if sha(ROOT/name)!=checksum:raise ValueError('Factorial input changed: '+name)
    result=json.loads((directory/'summary.json').read_text())
    if result['protocol']!=run['protocol'] or result['max_nominal_prediction_difference']>1e-7:
        raise ValueError('Unverified factorial study')
    v=result['variants']
    rows=[('Available','Enabled','full'),('Available','Disabled','no_rasch'),
          ('Removed','Enabled','no_count_prior'),('Removed','Disabled','no_count_prior_no_rasch')]
    for field,filename in [('brier','rasch_prior_table.tex'),('raw_brier','rasch_prior_raw_table.tex')]:
        table=[r'\begin{tabular}{llrr}',r'\toprule',r'Count prior & Rasch & $B_0$ & $B_1$\\',r'\midrule']
        for prior,rasch,key in rows:
            table.append(f'{prior} & {rasch} & '+ ' & '.join(f'{x:.5f}' for x in v[key][field])+r'\\')
        table += [r'\bottomrule',r'\end{tabular}']
        (ROOT/'report'/filename).write_text('\n'.join(table)+'\n')
    effect={condition:{field:[v[a][field][j]-v[b][field][j] for j in range(2)] for field in ['brier','raw_brier']}
            for condition,a,b in [('count_prior_available','full','no_rasch'),
                                 ('count_prior_removed','no_count_prior','no_count_prior_no_rasch')]}
    effect['definition']='With Rasch minus without Rasch; negative is better. Same subjects, items, labels, and frozen calibration.'
    effect['interaction']={field:[effect['count_prior_removed'][field][j]-effect['count_prior_available'][field][j]
                                  for j in range(2)] for field in ['brier','raw_brier']}
    (directory/'effects.json').write_text(json.dumps(effect,indent=2)+'\n')
    fallback=v['no_count_prior']['brier'];base=v['no_count_prior_no_rasch']['brier']
    # Render this interpretation only for the measured direction; do not recycle
    # a favorable explanation automatically if future runs change it.
    if not(all(a<b for a,b in zip(fallback,base)) and effect['count_prior_available']['brier'][1]>0):
        raise ValueError('Effect direction changed; revise the scientific interpretation')
    text=(r'\paragraph{Complementarity with the public count prior.} '
          +f"With the count prior removed, retaining Rasch lowers $B_0$ from ${base[0]:.5f}$ to ${fallback[0]:.5f}$ "
          +f"and $B_1$ from ${base[1]:.5f}$ to ${fallback[1]:.5f}$ (Table~\\ref{{tab:rasch_prior}}). "
          +r'With the count prior available, the zero-label benefit is smaller and the one-label effect is adverse. '
          +r'\textbf{Rasch supplies complementary low-budget information when the count-based signal is absent}; '
          +r'its marginal contribution depends on the retained priors and calibration. '
          +r'Pre-calibration comparisons in Appendix~\ref{app:prior_ablation} show that topic correction affects the size and sign of these gains. '
          +r'This supports retaining Rasch as part of the calibrated hybrid, without asserting that the fallback condition represents private competition data.')
    (ROOT/'report/rasch_prior_analysis.tex').write_text(text+'\n')
    nominal=effect['count_prior_available'];fallback_effect=effect['count_prior_removed']
    text=(r'The table reports the same factorial experiment before adding the topic correction. '
          +f"With the count prior available, Rasch changes raw $B_0$ by ${nominal['raw_brier'][0]:+.5f}$, "
          +f"versus ${nominal['brier'][0]:+.5f}$ after calibration. "
          +r'Thus the standard-prior zero-label benefit is conditional on the fixed calibration layer. '
          +f"Without the count prior, the raw zero-label difference is ${fallback_effect['raw_brier'][0]:+.5f}$, "
          +f"while the raw one-label difference is ${fallback_effect['raw_brier'][1]:+.5f}$; "
          +r'the latter is close to zero and becomes favorable after correction. '
          +r'These interactions preclude interpreting the final-score improvement as an isolated Rasch-only benefit. '
          +r'No component is refitted, and all conditions preserve the query metadata and binary subject prior; '
          +r'only the count-table lookup and Rasch mixing term are disabled. The study is not a held-out-source or missing-metadata benchmark.')
    (ROOT/'report/rasch_prior_raw_analysis.tex').write_text(text+'\n')
    print('Generated factorial prior table, interaction analysis, and uncalibrated appendix.')


if __name__=='__main__':main()
