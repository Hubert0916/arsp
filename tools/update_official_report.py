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
    metrics=summary['complete_six_budget'];n=summary['n_complete_blocks'];alc=metrics['brier_alc']
    def save(name,text): (report/name).write_text(text+'\n')
    save('official_metadata.tex',r'\newcommand{\OfficialRuns}{'+str(n)+'}')
    table=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',
           r'ASRP + Adaptive & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\',r'\midrule']
    for key,label in [('mean','Mean'),('sample_sd','SD')]:
        table.append(label+' & '+' & '.join(f'{m[key]:.4f}' for m in metrics['brier']+[alc])+r'\\')
    save('official_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    text=(r'Table~\ref{tab:official} summarizes \textbf{'+str(n)+' complete, distinct formative-feedback records} supplied for ASRP with adaptive acquisition. '
          +f"Their mean Brier ALC is ${alc['mean']:.4f}$ with sample SD ${alc['sample_sd']:.4f}$ and range ${alc['min']:.4f}$--${alc['max']:.4f}$. "
          +f"The mean Brier score falls from ${metrics['brier'][0]['mean']:.4f}$ at budget zero to ${metrics['brier'][-1]['mean']:.4f}$ at budget 31. "
          +r'These scores describe official formative samples, whereas the paired predictor/acquisition comparisons above use a fixed public proxy. No matched official Empirical Mean or Random-acquisition feedback is supplied, so these records do not establish an official advantage over either comparator. '
          +r'Cross-record SD describes variation in the supplied samples; it is not a confidence interval or a significance test.')
    save('official_overview.tex',text)
    save('official_abstract.tex',f"Separately, {n} complete, distinct official formative-feedback records for ASRP with adaptive acquisition have mean Brier ALC ${alc['mean']:.4f}$ (SD ${alc['sample_sd']:.4f}$).")
    save('official_conclusion.tex',f"The supplied official formative feedback gives ASRP with adaptive acquisition a mean Brier ALC of ${alc['mean']:.4f}$ across {n} complete, distinct records.")
    audit=json.loads((directory/'ingestion.json').read_text())
    unique_alc=summary['all_available_alc_summaries']
    duplicates=[r for r in runs if r['duplicate_of'] is not None]
    partial=[r for r in runs if not r['complete_six_budget'] and r['duplicate_of'] is None]
    note=(f"The supplied transcription contains {len(runs)} feedback headers and ALC summaries. "
          +f"Of these, {len(duplicates)} repeats earlier feedback and {len(partial)} lacks at least one complete budget table, leaving {n} complete, distinct records for Table~"+r'\ref{tab:official}. ')
    for r in duplicates:
        note+=f"Block {r['block']} reproduces block {r['duplicate_of']}'s group identities, counts, ALC, ECE, and all six budget scores, so it is counted once. "
    for r in partial:
        missing=[str(b) for b,v in zip(summary['budgets'],r['brier']) if v is None]
        note+=f"Block {r['block']} has a valid ALC summary but no complete table for budget(s) {', '.join(missing)}. "
    extras=sum(i.get('n_rows',0) for i in audit['issues'] if i['type']=='unassigned_rows')
    if extras: note+=f"Another {extras} rows cannot be assigned to an ALC summary and are excluded without reconstructing missing scores. "
    if unique_alc['n']!=n:
        note+=f"Using all {unique_alc['n']} distinct available ALC summaries, including the incomplete record, gives mean ALC ${unique_alc['mean']:.4f}$ (SD ${unique_alc['sample_sd']:.4f}$); the main six-budget table uses the same {n} complete records in every column. "
    note+=r'Block numbers indicate text order, not Codabench submission IDs. The transcription contains neither submission IDs nor archive hashes. The original feedback is retained locally; released aggregates omit anonymous group IDs.'
    save('official_audit.tex',note)
    table=[r'\begin{tabular}{lrrrl}',r'\toprule',r'Block & Groups & Eval. items & Brier ALC & Use in main table\\',r'\midrule']
    for r in runs:
        status=(f"Duplicate of {r['duplicate_of']}" if r['duplicate_of'] is not None else 'Included' if r['complete_six_budget'] else 'Missing budget(s) '+', '.join(str(b) for b,v in zip(summary['budgets'],r['brier']) if v is None))
        table.append(f"{r['block']} & {r['n_groups']} & {r['n_evaluation_items']} & {r['brier_alc']:.6f} & {status}"+r'\\')
    save('official_runs_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}']))
    print(f'Updated official report inputs from {n} complete, distinct records.')


if __name__=='__main__':main()
