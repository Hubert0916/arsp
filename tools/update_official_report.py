"""Export optional official-feedback fragments, separate from the public-only paper."""
import argparse
import json
import hashlib
from pathlib import Path
from parse_official_results import ROOT, summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/official/report_fragments')
    args=parser.parse_args()
    directory=ROOT/'results/official';report=args.output
    report.mkdir(parents=True,exist_ok=True)
    state=json.loads((directory/'status.json').read_text())
    digest=hashlib.sha256((ROOT/'artifacts/manifest.sha256.json').read_bytes()).hexdigest()
    if state['status']=='pending':
        if state['acquisition_policy']!='random' or state['inference_manifest_sha256']!=digest:
            raise ValueError('Pending official status does not match the current release')
        (report/'official_overview.tex').write_text('Official evaluation of the current ASRP submission with uniform random acquisition is pending.\n')
        (report/'official_limitation.tex').write_text(r'\paragraph{Transfer to private evaluation.} Frozen public priors, difficulty estimates, and topic corrections may transfer imperfectly to unseen subjects and benchmarks, especially before labels are available. Official results for the current random-acquisition submission are pending.'+'\n')
        for name in ['official_metadata.tex','official_table.tex','official_runs_table.tex','official_audit.tex']:
            (report/name).unlink(missing_ok=True)
        print('Official results pending for the current Random-acquisition release.')
        return
    if (state['status']!='complete' or state['model_version']!='rasch-hybrid-random-v1'
            or state['acquisition_policy']!='random'
            or state['inference_manifest_sha256']!=digest):
        raise ValueError('Official results must match the current Random-acquisition release')
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
           r'ASRP + Random & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\',r'\midrule']
    for key,label in [('mean','Mean'),('sample_sd','SD')]:
        table.append(label+' & '+' & '.join(f'{m[key]:.4f}' for m in metrics['brier']+[alc])+r'\\')
    save('official_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    text=(f"Across {n} official draws of the same ASRP + Random submission, "
          +r"\textbf{mean Brier ALC is $"+f"{alc['mean']:.4f}"+r"$} (sample SD $"+f"{alc['sample_sd']:.4f}"+r"$; Table~\ref{tab:official}). "
          +"These private-data results are separate from the public simulation.")
    save('official_overview.tex',text)
    save('official_limitation.tex',
         r'\paragraph{Cold-start transfer.} '
         +r'Frozen public priors, difficulty estimates, and topic corrections may transfer poorly before labels are available. '
         +f"Across {n} official evaluations, zero-label Brier is ${metrics['brier'][0]['mean']:.4f}$, "
         +r'above the $0.2500$ of a constant $0.5$ prediction. '
         +f"Revealed evidence reduces it to ${metrics['brier'][-1]['mean']:.4f}$ at budget 31. "
         +r'These aggregate scores do not identify which initialization component causes the private-data errors; accurate cold-start prediction remains a limitation.')
    note=(f"We evaluated the same ASRP + "
          +r"\texttt{random} ZIP in "
          +f"{n} official formative evaluations. " if same_zip else "")
    note+=r'Table~\ref{tab:official_records} lists the submission IDs and corresponding results.'
    save('official_audit.tex',note)
    table=[r'\begin{tabular}{lrrr}',r'\toprule',r'Submission ID & Groups & Eval. items & Brier ALC\\',r'\midrule']
    for r in sorted(runs,key=lambda r:by_block[r['block']]['submission_id']):
        table.append(f"{by_block[r['block']]['submission_id']} & {r['n_groups']} & {r['n_evaluation_items']} & {r['brier_alc']:.6f}"+r'\\')
    save('official_runs_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    print(f'Updated official report inputs from {n} complete, distinct records.')


if __name__=='__main__':main()
