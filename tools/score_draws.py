"""Audit and score all recorded evaluation responses, groups, then draws."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

from prepare_proxy import ROOT, sha, write_json
from draw_protocol import BUDGETS, WEIGHTS

METHODS=('full','no_rasch','no_kernel','no_topic','empirical_mean')


def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def describe(values):
    return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),
            'min':min(values),'max':max(values),'median':statistics.median(values)}


def ece(rows,variant,j):
    bins=defaultdict(list)
    for r in rows:bins[min(int(r[variant][j]*10),9)].append(r)
    return math.fsum(abs(math.fsum(r[variant][j]-r['y'] for r in cell)) for cell in bins.values())/len(rows)


def score(directory):
    run=json.loads((directory/'run.json').read_text())
    for name,checksum in run['inputs'].items():
        if sha(ROOT/name)!=checksum:raise ValueError('Input changed: '+name)
    for name,checksum in run['files'].items():
        if sha(directory/name)!=checksum:raise ValueError('Output checksum mismatch: '+name)
    manifest=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    config=manifest['config'];groups={g['group_id']:g for g in manifest['groups']}
    expected=set()
    for d in manifest['draws']:
        assert d['n_source_pairs']==sum(groups[g]['n_items'] for g in d['group_ids'])<=1000
        for policy in config['policies']:
            for gid in d['group_ids']:
                for item in groups[gid]['evaluation']:
                    for rid in item['response_ids']:expected.add((d['draw_id'],policy,gid,item['item_id'],rid))
    cells=defaultdict(list);seen=set()
    for r in read(directory/'predictions.jsonl.gz'):
        key=tuple(r[k] for k in ['draw_id','policy','group_id','item_id','response_id'])
        if key not in expected or key in seen:raise ValueError('Missing, duplicate, or acquired evaluation row')
        seen.add(key)
        if r['y'] not in (0,1):raise ValueError('Nonbinary outcome')
        for variant in METHODS:
            if len(r[variant])!=6 or any(not math.isfinite(p) or not 0<=p<=1 for p in r[variant]):raise ValueError('Invalid prediction')
        if r['empirical_mean'][0]!=.5:raise ValueError('Cold-start baseline mismatch')
        if r['full'][4:]!=r['no_rasch'][4:]:raise ValueError('Unused Rasch removal changed high-budget predictions')
        cells[(r['draw_id'],r['policy'],r['group_id'])].append(r)
    if seen!=expected:raise ValueError('Incomplete evaluation response coverage')
    by_group=[]
    for (draw,policy,gid),rows in sorted(cells.items()):
        g=groups[gid]
        entry={'draw_id':draw,'policy':policy,'group_id':gid,'source':g['source'],
               'subject_id':g['subject_id'],'n_items':len(g['evaluation']),'n_responses':len(rows),'methods':{}}
        for variant in METHODS:
            curve=[math.fsum((r[variant][j]-r['y'])**2 for r in rows)/len(rows) for j in range(6)]
            entry['methods'][variant]={'brier':curve,'alc':math.fsum(w*b for w,b in zip(WEIGHTS,curve)),
                                       'ece':[ece(rows,variant,j) for j in range(6)]}
        by_group.append(entry)
    by_draw=[]
    for d in manifest['draws']:
        for policy in config['policies']:
            rows=[r for r in by_group if r['draw_id']==d['draw_id'] and r['policy']==policy]
            if len(rows)!=len(d['group_ids']):raise ValueError('Group count mismatch')
            by_draw.append({**d,'policy':policy,'n_groups':len(rows),
                'n_eval_items':sum(r['n_items'] for r in rows),'n_eval_responses':sum(r['n_responses'] for r in rows),
                'methods':{v:{'brier':[statistics.mean(r['methods'][v]['brier'][j] for r in rows) for j in range(6)],
                              'alc':statistics.mean(r['methods'][v]['alc'] for r in rows)} for v in METHODS}})
    methods={}
    for policy in config['policies']:
        rows=[r for r in by_draw if r['policy']==policy]
        methods[policy]={v:{'brier':[describe([r['methods'][v]['brier'][j] for r in rows]) for j in range(6)],
                            'alc':describe([r['methods'][v]['alc'] for r in rows])} for v in METHODS}
    paired={}
    index={(r['draw_id'],r['policy']):r['methods'] for r in by_draw}
    comparisons=[('predictor_adaptive','adaptive','full','adaptive','empirical_mean'),
                 ('predictor_random','random','full','random','empirical_mean'),
                 ('acquisition_asrp','adaptive','full','random','full')]
    comparisons += [('remove_'+v,'adaptive',v,'adaptive','full') for v in ('no_rasch','no_kernel','no_topic')]
    for name,pa,va,pb,vb in comparisons:
        deltas=[index[(d['draw_id'],pa)][va]['alc']-index[(d['draw_id'],pb)][vb]['alc'] for d in manifest['draws']]
        paired[name]={'definition':f'{pa}/{va} minus {pb}/{vb}',**describe(deltas),
            'n_negative':sum(x<0 for x in deltas),'n_positive':sum(x>0 for x in deltas),'n_zero':sum(x==0 for x in deltas),
            'brier_difference':[statistics.mean(index[(d['draw_id'],pa)][va]['brier'][j]-index[(d['draw_id'],pb)][vb]['brier'][j] for d in manifest['draws']) for j in range(6)]}
    acquisition=[]
    traces=read(directory/'acquisition.jsonl.gz')
    if {(t['draw_id'],t['policy']) for t in traces}!=set(index) or len(traces)!=len(index):raise ValueError('Missing trajectory')
    for t in traces:
        accepted=defaultdict(list);past=set();seen_count=0;last_budget=0
        for r in t['decisions']:
            g=groups[r['group_id']]
            if r['budget']<last_budget or r['evidence_before']!=seen_count:raise ValueError('Evidence ordering mismatch')
            last_budget=r['budget']
            pool=g['acquisition'];slot=pool[r['position']]
            if slot['item_id']!=r['item_id'] or (r['group_id'],r['item_id']) in past:raise ValueError('Invalid candidate')
            past.add((r['group_id'],r['item_id']))
            if r['acquired']:
                if r['response_id']!=slot['response_id']:raise ValueError('Wrong acquired response')
                accepted[r['group_id']].append(r);seen_count+=1
        if any(len(accepted[g])!=31 for g in t['groups']):raise ValueError('Incomplete acquisition')
        acquisition.append({'draw_id':t['draw_id'],'policy':t['policy'],
            'n_candidates_seen':len(t['decisions']),'n_labels':seen_count,
            'n_groups':len(t['groups'])})
    all_draws=[d for d in by_draw if d['policy']=='adaptive']
    summary={'protocol':config,'n_draws':len(manifest['draws']),
        'n_eligible_groups':len(manifest['eligible_groups']),
        'n_eligible_sources':sum(i['eligible_groups']>0 for i in manifest['inventory']),
        'n_distinct_groups':len(groups),'n_distinct_sources':len({g['source'] for g in groups.values()}),
        'n_distinct_subject_aliases':len({g['subject_alias'] for g in groups.values()}),
        'group_occurrences':sum(d['n_groups'] for d in all_draws),
        'group_count_range':[min(d['n_groups'] for d in all_draws),max(d['n_groups'] for d in all_draws)],
        'source_pair_range':[min(d['n_source_pairs'] for d in all_draws),max(d['n_source_pairs'] for d in all_draws)],
        'n_eval_item_occurrences_per_policy':sum(d['n_eval_items'] for d in all_draws),
        'n_eval_response_occurrences_per_policy':sum(d['n_eval_responses'] for d in all_draws),
        'methods':methods,'paired':paired,
        'scope':'Fixed public snapshot; SD describes group-composition variability, not an independent private-source generalization estimate.'}
    for name,value in [('summary.json',summary),('by_group.json',by_group),('draws.json',by_draw),('acquisition_summary.json',acquisition)]:write_json(directory/name,value)
    print(json.dumps({k:summary[k] for k in ('n_draws','n_distinct_groups','n_distinct_sources','group_count_range','source_pair_range','n_eval_item_occurrences_per_policy','n_eval_response_occurrences_per_policy')},indent=2))
    for policy in methods:
        for v in METHODS:print(policy,v,methods[policy][v]['alc'])
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path,nargs='?',default=ROOT/'results/formative')
    score(p.parse_args().directory)
