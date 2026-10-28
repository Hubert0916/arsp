"""Generate official-feedback report tables from audited evaluation aggregates."""
import json
from pathlib import Path
from parse_official_results import ROOT, summarize


def main():
    directory=ROOT/'results/official';report=ROOT/'report'
    runs=json.loads((directory/'runs.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if summarize(runs)!=summary:
        raise ValueError('Official summary differs from released per-evaluation aggregates')
    provenance=json.loads((directory/'provenance.json').read_text())
    submissions=json.loads((directory/'submissions.json').read_text())['submissions']
    by_block={s['feedback_block']:s for s in submissions}
    if set(by_block)!={r['block'] for r in runs} or len({s['submission_id'] for s in submissions})!=len(runs):
        raise ValueError('Submission IDs must map one-to-one to feedback records')
    for run in runs:
        if abs(run['brier_alc']-by_block[run['block']]['displayed_brier_alc'])>1.00001e-6:
            raise ValueError('Submission screenshot score disagrees with reconstructed ALC')
    same_zip=provenance.get('same_submission_zip_confirmed_by_submitter',False)
    metrics=summary['complete_six_budget'];n=summary['n_complete_blocks'];alc=metrics['brier_alc']
    def save(name,text): (report/name).write_text(text+'\n')
    save('official_metadata.tex',r'\newcommand{\OfficialRuns}{'+str(n)+'}')
    table=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',
           r'ASRP + Adaptive & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\',r'\midrule']
    for key,label in [('mean','Mean'),('sample_sd','SD')]:
        table.append(label+' & '+' & '.join(f'{m[key]:.4f}' for m in metrics['brier']+[alc])+r'\\')
    save('official_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    text=(f"Across {n} official formative evaluations of the same ASRP submission with adaptive acquisition, "
          +r"\textbf{mean Brier ALC is $"+f"{alc['mean']:.4f}"+r"$} (sample SD $"+f"{alc['sample_sd']:.4f}"+r"$; Table~\ref{tab:official}). "
          +f"Mean Brier score decreases from ${metrics['brier'][0]['mean']:.4f}$ at budget zero to ${metrics['brier'][-1]['mean']:.4f}$ at budget 31. "
          +"These official samples are separate from the local public-proxy comparisons.")
    save('official_overview.tex',text)
    save('official_limitation.tex',
         r'\paragraph{Cold-start transfer.} '
         +r'Without revealed labels, ASRP relies on frozen public-data priors, estimated item difficulty, and topic corrections. '
         +f"Across the {n} official evaluations, its mean zero-label Brier score is ${metrics['brier'][0]['mean']:.4f}$, "
         +r'compared with $0.2500$ for a constant prediction of $0.5$ on any binary outcome. '
         +r'This identifies a weakness in the combined initialization on the sampled private data; the aggregate scores do not identify which component is responsible. '
         +f"Revealed evidence reduces mean Brier to ${metrics['brier'][-1]['mean']:.4f}$ at budget 31, but accurate prediction before acquiring labels remains a limitation.")
    save('official_conclusion.tex',f"The supplied official formative feedback gives ASRP with adaptive acquisition a mean Brier ALC of ${alc['mean']:.4f}$ across {n} complete, distinct records.")
    note=(f"We evaluated the same ASRP + "
          +r"\texttt{adaptive\_4\_warm} ZIP in "
          +f"{n} official formative evaluations. " if same_zip else "")
    note+=r'Table~\ref{tab:official_records} lists the submission IDs and corresponding results.'
    save('official_audit.tex',note)
    table=[r'\begin{tabular}{lrrr}',r'\toprule',r'Submission ID & Groups & Eval. items & Brier ALC\\',r'\midrule']
    for r in sorted(runs,key=lambda r:by_block[r['block']]['submission_id']):
        table.append(f"{by_block[r['block']]['submission_id']} & {r['n_groups']} & {r['n_evaluation_items']} & {r['brier_alc']:.6f}"+r'\\')
    save('official_runs_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    print(f'Updated official report inputs from {n} complete, distinct records.')


if __name__=='__main__':main()
