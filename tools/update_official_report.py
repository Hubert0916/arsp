"""Generate official-feedback report tables from audited evaluation aggregates."""
import json
import hashlib
import statistics
from pathlib import Path
from parse_official_results import ROOT, summarize


def main():
    directory=ROOT/'results/official';report=ROOT/'report'
    state=json.loads((directory/'status.json').read_text())
    digest=hashlib.sha256((ROOT/'artifacts/manifest.sha256.json').read_bytes()).hexdigest()
    if (state['status']!='complete' or state['model_version']!='rasch-hybrid-v1'
            or state['inference_manifest_sha256']!=digest):
        raise ValueError('Official results must be associated with the restored inference version')
    runs=json.loads((directory/'runs.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if summarize(runs)!=summary:
        raise ValueError('Official summary differs from released per-evaluation aggregates')
    provenance=json.loads((directory/'provenance.json').read_text())
    if (provenance.get('model_version') != state['model_version']
            or not provenance.get('applies_to_current_runtime')
            or not provenance.get('same_submission_zip_confirmed_by_submitter')):
        raise ValueError('Official feedback provenance does not match the retained model')
    submissions=json.loads((directory/'submissions.json').read_text())['submissions']
    by_block={s['feedback_block']:s for s in submissions}
    if set(by_block)!={r['block'] for r in runs} or len({s['submission_id'] for s in submissions})!=len(runs):
        raise ValueError('Submission IDs must map one-to-one to feedback records')
    for run in runs:
        if abs(run['brier_alc']-by_block[run['block']]['displayed_brier_alc'])>1.00001e-6:
            raise ValueError('Submission screenshot score disagrees with reconstructed ALC')
    same_zip=provenance.get('same_submission_zip_confirmed_by_submitter',False)
    metrics=summary['complete_six_budget'];n=summary['n_complete_blocks'];alc=metrics['brier_alc']
    if n!=state['n_evaluations']:
        raise ValueError('Official evaluation count differs from the verified feedback')
    def save(name,text): (report/name).write_text(text+'\n')
    macros={'OfficialRuns':str(n), 'OfficialALC':f"{alc['mean']:.4f}",
            'OfficialZeroBrier':f"{metrics['brier'][0]['mean']:.4f}"}
    save('official_metadata.tex','\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items()))
    table=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',
           r'ASRP + Adaptive & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\',r'\midrule']
    for key,label in [('mean','Mean'),('sample_sd','SD')]:
        table.append(label+' & '+' & '.join(f'{m[key]:.4f}' for m in metrics['brier']+[alc])+r'\\')
    save('official_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    text=(f"Across {n} official formative evaluations of the same ASRP submission with adaptive acquisition, "
          +r"\textbf{mean Brier ALC is $"+f"{alc['mean']:.4f}"+r"$} (sample SD $"+f"{alc['sample_sd']:.4f}"+r"$; Table~\ref{tab:official}). "
          +"These official samples are separate from the local public-proxy comparisons.")
    save('official_overview.tex',text)
    save('official_limitation.tex',
         r'\paragraph{Cold-start transfer.} '
         +r'Frozen public priors, difficulty estimates, and topic corrections may transfer poorly before labels are available. '
         +f"Across {n} official evaluations, zero-label Brier is ${metrics['brier'][0]['mean']:.4f}$, "
         +r'above the $0.2500$ of a constant $0.5$ prediction. '
         +f"Revealed evidence reduces it to ${metrics['brier'][-1]['mean']:.4f}$ at budget 31. "
         +r'The public ablations do not identify which initialization component causes the private-data errors; accurate cold-start prediction remains a limitation.')
    note=(f"We evaluated the same ASRP + "
          +r"\texttt{adaptive\_4\_warm} ZIP in "
          +f"{n} official formative evaluations. " if same_zip else "")
    note+=r'Table~\ref{tab:official_records} lists the submission IDs and corresponding results.'
    save('official_audit.tex',note)
    table=[r'\begin{tabular}{lrrr}',r'\toprule',r'Submission ID & Groups & Eval. items & Brier ALC\\',r'\midrule']
    for r in sorted(runs,key=lambda r:by_block[r['block']]['submission_id']):
        table.append(f"{by_block[r['block']]['submission_id']} & {r['n_groups']} & {r['n_evaluation_items']} & {r['brier_alc']:.6f}"+r'\\')
    save('official_runs_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    comparison=json.loads((directory/'without_rasch_screenshot.json').read_text())
    entries=comparison['submissions']
    if len({r['submission_id'] for r in entries})!=len(entries):
        raise ValueError('Duplicate screenshot submission ID')
    included=[r for r in entries if r['status_in_screenshot']=='Finished']
    scores=[r['displayed_brier_alc'] for r in included]
    if len(scores)<2 or any(not 0<=v<=1 for v in scores):
        raise ValueError('Invalid completed screenshot scores')
    new={'n':len(scores),'mean':statistics.mean(scores),'sample_sd':statistics.stdev(scores),
         'min':min(scores),'max':max(scores)}
    aggregate={'with_rasch':alc,'without_rasch':new,
               'difference_without_minus_with':new['mean']-alc['mean'],
               'included_without_rasch_ids':[r['submission_id'] for r in included],
               'excluded_without_rasch_ids':[r['submission_id'] for r in entries if r not in included],
               'design':'Unpaired official batches from different dates; not a controlled Rasch ablation.',
               'without_rasch_budget_scores':None}
    (directory/'comparison.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    table=[r'\begin{tabular}{lrrr}',r'\toprule',r'Official batch & Runs & Mean ALC & SD\\',r'\midrule',
           f"With Rasch & {n} & {alc['mean']:.5f} & {alc['sample_sd']:.5f}"+r'\\',
           f"Without Rasch & {new['n']} & {new['mean']:.5f} & {new['sample_sd']:.5f}"+r'\\',
           r'\bottomrule',r'\end{tabular}']
    save('official_comparison_table.tex','\n'.join(table))
    text=(f"The {new['n']} completed no-Rasch submissions in the later platform screenshot have mean ALC "
          +f"${new['mean']:.5f}$ (SD ${new['sample_sd']:.5f}$), compared with ${alc['mean']:.5f}$ "
          +f"(SD ${alc['sample_sd']:.5f}$) for the {n} with-Rasch evaluations. "
          +r'The cancelled submission is excluded. These batches use different random draws and dates; '
          +r'only total ALC is available for the no-Rasch batch. '
          +r'\textbf{This unpaired comparison} motivates retaining the prior submission, '
          +r'but cannot identify Rasch as the cause or locate the difference by label budget.')
    save('official_comparison_analysis.tex',text)
    table=[r'\begin{tabular}{rr@{\qquad}rr}',r'\toprule',
           r'Submission ID & Brier ALC & Submission ID & Brier ALC\\',r'\midrule']
    ordered=sorted(included,key=lambda r:r['submission_id']);half=(len(ordered)+1)//2
    for index,left in enumerate(ordered[:half]):
        cells=[str(left['submission_id']),f"{left['displayed_brier_alc']:.6f}"]
        right=ordered[index+half] if index+half<len(ordered) else None
        cells += [str(right['submission_id']),f"{right['displayed_brier_alc']:.6f}"] if right else ['','']
        table.append(' & '.join(cells)+r'\\')
    save('official_without_rasch_runs.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    print(f'Updated official report inputs from {n} complete, distinct records.')


if __name__=='__main__':main()
