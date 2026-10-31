"""Audit copied Codabench feedback and export per-evaluation aggregates.

Anonymous group IDs are used locally to validate table joins; public outputs
contain only evaluation-level aggregates. Incomplete or unassigned rows are
recorded explicitly, never attached to another evaluation or imputed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [.1, .2, .2, .2, .2, .1]
ROW = re.compile(r'(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')
ROUNDING_TOLERANCE = 1.00001e-6


def parse(text):
    blocks, current, table = [], None, None
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line == 'Formative feedback':
            current = {'block': len(blocks)+1, 'line_start': line_number, 'tables': []}
            blocks.append(current)
            table = None
        elif line == 'ALC summary' or re.fullmatch(r'Label budget: \d+', line):
            if current is None:
                raise ValueError(f'Line {line_number}: table outside feedback block')
            section = 'alc' if line == 'ALC summary' else int(line.split(':')[1])
            if section != 'alc' and section not in BUDGETS:
                raise ValueError(f'Unexpected budget at line {line_number}')
            table = {'section': section, 'line': line_number, 'rows': []}
            current['tables'].append(table)
        elif match := ROW.fullmatch(line):
            if table is None:
                raise ValueError(f'Line {line_number}: row outside table')
            subject, benchmark, count, metric, ece = match.groups()
            row = {'key': (subject, benchmark), 'count': int(count),
                   'metric': float(metric), 'ece': float(ece), 'line': line_number}
            if row['count'] <= 0 or not all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ('metric', 'ece')):
                raise ValueError(f'Invalid score/count at line {line_number}')
            table['rows'].append(row)
        elif line and not line.startswith('subject_id') and not re.fullmatch(r'[-\s]+', line):
            raise ValueError(f'Unrecognized content at line {line_number}')
    if not blocks:
        raise ValueError('No formative feedback found')
    runs, issues, fingerprints = [], [], {}
    for block in blocks:
        summaries = [t for t in block['tables'] if t['section'] == 'alc']
        if len(summaries) != 1 or not summaries[0]['rows']:
            raise ValueError(f"Block {block['block']}: expected one nonempty ALC summary")
        summary = {r['key']: r for r in summaries[0]['rows']}
        if len(summary) != len(summaries[0]['rows']):
            raise ValueError(f"Duplicate summary group in block {block['block']}")
        joined = {}
        for budget in BUDGETS:
            rows = {}
            for tab in [t for t in block['tables'] if t['section'] == budget]:
                foreign = []
                for row in tab['rows']:
                    key = row['key']
                    if key not in summary:
                        foreign.append(row['line'])
                        continue
                    if key in rows:
                        raise ValueError(f"Duplicate group/budget at line {row['line']}; refusing to merge")
                    if row['count'] != summary[key]['count']:
                        raise ValueError(f"Changed evaluation count at line {row['line']}")
                    rows[key] = row
                if foreign:
                    issues.append({'block': block['block'], 'type': 'unassigned_rows',
                                   'preceding_budget_header': budget, 'line_start': min(foreign),
                                   'line_end': max(foreign), 'n_rows': len(foreign),
                                   'note': 'Group keys absent from this ALC summary; no run/budget inferred.'})
            if set(rows) != set(summary):
                issues.append({'block': block['block'], 'type': 'incomplete_budget',
                               'budget': budget, 'expected_groups': len(summary), 'observed_groups': len(rows)})
            joined[budget] = rows
        complete = all(set(joined[b]) == set(summary) for b in BUDGETS)
        max_alc_difference = max_ece_difference = None
        if complete:
            max_alc_difference = max(abs(math.fsum(w*joined[b][key]['metric'] for w,b in zip(WEIGHTS,BUDGETS))-summary[key]['metric']) for key in summary)
            max_ece_difference = max(abs(math.fsum(w*joined[b][key]['ece'] for w,b in zip(WEIGHTS,BUDGETS))-summary[key]['ece']) for key in summary)
            if max(max_alc_difference,max_ece_difference) > ROUNDING_TOLERANCE:
                raise ValueError(f"Block {block['block']}: summary disagrees with budget-weighted scores")
        canonical = [(key, summary[key]['count'], summary[key]['metric'], summary[key]['ece'],
                      [(b, joined[b].get(key,{}).get('metric'), joined[b].get(key,{}).get('ece')) for b in BUDGETS]) for key in sorted(summary)]
        fingerprint = hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
        duplicate_of = fingerprints.get(fingerprint)
        if duplicate_of is not None:
            issues.append({'block': block['block'], 'type': 'duplicate_feedback', 'duplicate_of': duplicate_of,
                           'note': 'Identical group identities, counts, ALC, ECE, and all available budget scores; counted once.'})
        else:
            fingerprints[fingerprint] = block['block']
        runs.append({'block': block['block'], 'line_start': block['line_start'],
                     'n_groups': len(summary), 'n_evaluation_items': sum(r['count'] for r in summary.values()),
                     'complete_six_budget': complete, 'duplicate_of': duplicate_of,
                     'brier': [statistics.mean(r['metric'] for r in joined[b].values()) if set(joined[b])==set(summary) else None for b in BUDGETS],
                     'ece': [statistics.mean(r['ece'] for r in joined[b].values()) if set(joined[b])==set(summary) else None for b in BUDGETS],
                     'brier_alc': statistics.mean(r['metric'] for r in summary.values()),
                     'weighted_ece': statistics.mean(r['ece'] for r in summary.values()),
                     'max_group_alc_rounding_difference': max_alc_difference,
                     'max_group_ece_rounding_difference': max_ece_difference})
    return runs, issues


def describe(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values) if len(values)>1 else None,
            'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def summarize(runs):
    unique = [r for r in runs if r['duplicate_of'] is None]
    complete = [r for r in unique if r['complete_six_budget']]
    if not complete:
        raise ValueError('No complete six-budget feedback block')
    return {'method': 'ASRP (acquisition policy recorded separately in provenance)',
            'budgets': BUDGETS, 'alc_weights': WEIGHTS,
            'weighting': 'Equal groups within each evaluation; equal evaluations across included blocks.',
            'precision': 'Aggregated from the six-decimal scores in copied platform feedback.',
            'complete_blocks': [r['block'] for r in complete],
            'n_complete_blocks': len(complete), 'n_available_alc_summaries': len(unique),
            'n_raw_alc_summaries': len(runs), 'n_duplicate_blocks': len(runs)-len(unique),
            'complete_six_budget': {'brier': [describe([r['brier'][j] for r in complete]) for j in range(6)],
                                    'brier_alc': describe([r['brier_alc'] for r in complete]),
                                    'weighted_ece': describe([r['weighted_ece'] for r in complete])},
            'all_available_alc_summaries': describe([r['brier_alc'] for r in unique]),
            'groups_per_complete_evaluation': [min(r['n_groups'] for r in complete),max(r['n_groups'] for r in complete)],
            'evaluation_items_per_complete_evaluation': [min(r['n_evaluation_items'] for r in complete),max(r['n_evaluation_items'] for r in complete)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/official')
    args=p.parse_args()
    raw=args.input.read_bytes();runs,issues=parse(raw.decode('utf-8-sig'))
    summary=summarize(runs)
    audit={'input_sha256':hashlib.sha256(raw).hexdigest(), 'input_bytes':len(raw),
           'n_formative_headers':len(runs),'n_complete_blocks':summary['n_complete_blocks'],
           'issues':issues,'duplicate_blocks':summary['n_duplicate_blocks'],
           'group_alc_and_ece_rounding_tolerance':ROUNDING_TOLERANCE,
           'submission_ids_present':False,'archive_hash_present':False}
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in [('runs.json',runs),('summary.json',summary),('ingestion.json',audit)]:
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'complete_blocks':summary['complete_blocks'],
                      'complete_alc':summary['complete_six_budget']['brier_alc'],
                      'all_available_alc':summary['all_available_alc_summaries'],'issues':issues},indent=2))


if __name__=='__main__':main()
