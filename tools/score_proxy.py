"""Audit and aggregate the paired 2x2, multi-seed public evaluation."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from prepare_proxy import ROOT, sha, write_json

METHODS = ('asrp', 'empirical_mean')


def source_comparisons(by_group, seeds):
    """Compare paired source means, counting each source once across seeds."""
    cells = defaultdict(dict)
    for row in by_group:
        key = (row['source'], row['subject_id'])
        identity = (row['method'], row['policy'], row['seed'])
        if identity in cells[key]:
            raise ValueError('Duplicate group metric')
        cells[key][identity] = row['alc']
    pairs = {
        'predictor_random': (('asrp', 'random'), ('empirical_mean', 'random')),
        'predictor_adaptive': (('asrp', 'adaptive'), ('empirical_mean', 'adaptive')),
        'acquisition_asrp': (('asrp', 'adaptive'), ('asrp', 'random')),
    }
    result = {}
    for name, (left, right) in pairs.items():
        differences = []
        for (source, subject), values in sorted(cells.items()):
            try:
                delta = statistics.mean(values[(*left, seed)] - values[(*right, seed)] for seed in seeds)
            except KeyError as error:
                raise ValueError('Missing paired source/seed metric') from error
            differences.append({'source': source, 'subject_id': subject, 'delta_alc': delta})
        result[name] = {
            'left': '_'.join(left), 'right': '_'.join(right),
            'difference': 'left minus right; mean of paired seed differences',
            'tie_tolerance': 1e-12,
            'improved': sum(r['delta_alc'] < -1e-12 for r in differences),
            'tied': sum(abs(r['delta_alc']) <= 1e-12 for r in differences),
            'worsened': sum(r['delta_alc'] > 1e-12 for r in differences),
            'mean_delta_alc': statistics.mean(r['delta_alc'] for r in differences),
        }
    return result


def aggregate(records, config):
    cells = defaultdict(list)
    seen = set()
    for row in records:
        key = (row['source'], row['subject_id'], row['seed'], row['policy'])
        identity = (*key, row['item_id'])
        if identity in seen:
            raise ValueError('Duplicate evaluation item')
        seen.add(identity)
        if row['y'] not in (0, 1):
            raise ValueError('Non-binary evaluation outcome')
        for method in METHODS:
            if len(row[method]) != len(config['budgets']) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row[method]):
                raise ValueError('Invalid prediction array')
        cells[key].append(row)
    if not cells:
        raise ValueError('No evaluation records')
    by_group = []
    for (source, subject, seed, policy), rows in sorted(cells.items()):
        for method in METHODS:
            curve = [math.fsum((r[method][j]-r['y'])**2 for r in rows)/len(rows) for j in range(6)]
            by_group.append({'source': source, 'subject_id': subject, 'seed': seed, 'policy': policy,
                             'method': method, 'n_rows': len(rows), 'brier': curve,
                             'alc': math.fsum(w*b for w,b in zip(config['alc_weights'],curve))})
    per_seed, combinations = [], {}
    for method in METHODS:
        for policy in config['policies']:
            subset = [r for r in by_group if r['method']==method and r['policy']==policy]
            seed_results = []
            for seed in config['seeds']:
                rows = [r for r in subset if r['seed']==seed]
                if not rows:
                    raise ValueError('Missing seed/policy combination')
                values = {'method': method, 'policy': policy, 'seed': seed, 'n_groups': len(rows),
                          'brier': [math.fsum(r['brier'][j] for r in rows)/len(rows) for j in range(6)],
                          'alc': math.fsum(r['alc'] for r in rows)/len(rows)}
                seed_results.append(values)
                per_seed.append(values)
            combinations[f'{method}_{policy}'] = {
                'method': method, 'policy': policy,
                'brier': [statistics.mean(r['brier'][j] for r in seed_results) for j in range(6)],
                'alc': statistics.mean(r['alc'] for r in seed_results),
                'alc_seed_sd': statistics.stdev(r['alc'] for r in seed_results) if len(seed_results)>1 else 0.0,
                'per_seed_alc': {str(r['seed']):r['alc'] for r in seed_results}}
    effects = {}
    for method in METHODS:
        diffs = {str(seed): combinations[f'{method}_adaptive']['per_seed_alc'][str(seed)] - combinations[f'{method}_random']['per_seed_alc'][str(seed)] for seed in config['seeds']}
        effects[method] = {'adaptive_minus_random_alc': statistics.mean(diffs.values()),
                           'per_seed': diffs, 'seeds_improved': sum(v<0 for v in diffs.values())}
    return {'combinations': combinations, 'per_seed': per_seed, 'acquisition_effect': effects,
            'source_comparisons': source_comparisons(by_group, config['seeds'])}, by_group


def audit(records, acquisitions, manifest):
    config = manifest['config']
    groups = {(g['source'],g['subject_id']):g for g in manifest['groups']}
    trajectories = {}
    for a in acquisitions:
        key = (a['source'],a['subject_id'],a['seed'],a['policy'])
        if key in trajectories:
            raise ValueError('Duplicate acquisition trajectory')
        trajectories[key]=a
    expected_keys = {(s,t,seed,policy) for s,t in groups for seed in config['seeds'] for policy in config['policies']}
    if set(trajectories)!=expected_keys:
        raise ValueError('Missing or unexpected acquisition trajectory')
    truths, expected = {}, {}
    for key,a in trajectories.items():
        source,subject,seed,policy=key
        split=groups[(source,subject)]['splits'][str(seed)]
        pool=[r['item_id'] for r in split['acquisition']]
        targets={r['item_id']:r['response_id'] for r in split['evaluation']}
        if set(pool)&set(targets):
            raise ValueError('Acquisition and evaluation pools overlap')
        if a['pool_size']!=len(pool) or a['evaluation_size']!=len(targets) or a['n_items']!=len(pool)+len(targets):
            raise ValueError('Wrong pool sizes')
        ids,labels,decisions=a['acquired_item_ids'],a['labels'],a['decisions']
        if len(ids)!=31 or len(set(ids))!=31 or not set(ids)<=set(pool) or len(labels)!=31 or any(y not in (0,1) for y in labels):
            raise ValueError('Invalid acquired labels/items')
        if [d['position'] for d in decisions]!=list(range(len(decisions))):
            raise ValueError('Candidate order changed')
        if any(d['item_id']!=pool[d['position']] for d in decisions):
            raise ValueError('Candidate identity changed')
        if [d['item_id'] for d in decisions if d['acquired']]!=ids:
            raise ValueError('Decision log and acquired IDs disagree')
        acquired=0
        warmup=min(16,len(pool)-31)
        for d in decisions:
            reason=('forced' if len(pool)-d['position']<=31-acquired else 'warmup' if policy=='adaptive' and d['position']<warmup else policy)
            if d['reason']!=reason or (reason=='warmup' and d['acquired']) or (reason=='forced' and not d['acquired']):
                raise ValueError('Incorrect warm-up/forced-acquisition accounting')
            acquired+=int(d['acquired'])
        for item,response in targets.items():expected[(*key,item)]=response
    actual={}
    for row in records:
        key=(row['source'],row['subject_id'],row['seed'],row['policy'])
        full=(*key,row['item_id'])
        if full in actual:raise ValueError('Duplicate prediction row')
        actual[full]=row['response_id']
        truthkey=(row['source'],row['subject_id'],row['item_id'])
        truth=(row['response_id'],row['y'])
        if truths.setdefault(truthkey,truth)!=truth:raise ValueError('Evaluation outcomes differ across policies/seeds')
        labels=trajectories[key]['labels']
        if row['empirical_mean']!=[sum(labels[:n])/n if n else .5 for n in config['budgets']]:
            raise ValueError('Empirical Mean did not receive matched evidence')
    if actual!=expected:raise ValueError('Prediction coverage differs from split manifest')
    # Both policies must have exactly the same cold-start predictions.
    lookup={(r['source'],r['subject_id'],r['seed'],r['policy'],r['item_id']):r for r in records}
    for row in records:
        if row['policy']=='adaptive':
            other=lookup[(row['source'],row['subject_id'],row['seed'],'random',row['item_id'])]
            if abs(row['asrp'][0]-other['asrp'][0])>1e-12:raise ValueError('Cold-start prediction depends on acquisition policy')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=ROOT/'results/reproduced')
    parser.add_argument('--compare',type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'experiments/proxy_manifest.json').read_text());config=manifest['config']
    run=json.loads((args.directory/'run.json').read_text())
    if run['inference_manifest_sha256'] != sha(ROOT/'artifacts/manifest.sha256.json'):
        raise ValueError('Run belongs to different frozen inference files; use its matching Git revision')
    if run['config']!=config or run['split_manifest_sha256']!=sha(ROOT/'experiments/proxy_manifest.json') or run['data_sha256']!=manifest['prepared_sha256'] or run['n_groups']!=len(manifest['groups']):
        raise ValueError('Run does not match the complete published protocol')
    for name,expected in run['files'].items():
        if sha(args.directory/name)!=expected:raise ValueError('Result checksum mismatch: '+name)
    records=[json.loads(l) for l in (args.directory/'predictions.jsonl').read_text().splitlines()]
    acquisitions=[json.loads(l) for l in (args.directory/'acquisition.jsonl').read_text().splitlines()]
    audit(records,acquisitions,manifest)
    result,by_group=aggregate(records,config)
    large={g['source'] for g in manifest['groups'] if g['n_items']>=config['large_group_min_items']}
    result['at_least_160_items'],_=aggregate([r for r in records if r['source'] in large],config)
    result['at_least_160_items']['n_groups']=len(large)
    sizes=[g['n_items'] for g in manifest['groups']]
    decisions=Counter(d['reason'] for a in acquisitions if a['policy']=='adaptive' for d in a['decisions'])
    result.update(status='public_proxy',protocol_version=2,
                  model_version=json.loads((ROOT/'artifacts/asrp/config.json').read_text()).get('version', 'rasch-hybrid-v1'),
                  inference_manifest_sha256=run['inference_manifest_sha256'],
                  n_sources=len(manifest['groups']),n_groups=len(manifest['groups']),
                  n_items=sum(sizes),n_rows_per_seed=sum(n//2 for n in sizes),n_prediction_records=len(records),
                  group_items_range=[min(sizes),max(sizes)],acquisition_pool_range=[min((n+1)//2 for n in sizes),max((n+1)//2 for n in sizes)],
                  n_excluded_sources=sum(x['status']!='included' for x in manifest['source_inventory']),
                  seeds=config['seeds'],budgets=config['budgets'],dataset_revision=config['revision'],
                  split_manifest_sha256=sha(ROOT/'experiments/proxy_manifest.json'),
                  adaptive_decisions=dict(decisions),
                  adaptive_trajectories_with_discretion=sum(any(d['reason']=='adaptive' for d in a['decisions']) for a in acquisitions if a['policy']=='adaptive'))
    write_json(args.directory/'summary.json',result);write_json(args.directory/'by_source.json',by_group)
    if args.compare:
        reference=[json.loads(l) for l in (args.compare/'predictions.jsonl').read_text().splitlines()]
        if len(reference)!=len(records):raise ValueError('Reference row count differs')
        for a,b in zip(records,reference):
            for k in ['source','subject_id','seed','policy','item_id','response_id','y']:
                if a[k]!=b[k]:raise ValueError('Reference identities differ')
            for method in METHODS:
                if any(abs(p-q)>1e-7 for p,q in zip(a[method],b[method])):raise ValueError('Predictions differ by more than 1e-7')
        if (args.directory/'acquisition.jsonl').read_bytes()!=(args.compare/'acquisition.jsonl').read_bytes():raise ValueError('Reference acquisition differs')
        print('All predictions match within 1e-7; acquisition trajectories match exactly.')
    print(json.dumps({k:result[k] for k in ['n_groups','n_rows_per_seed','combinations','acquisition_effect','adaptive_decisions']},indent=2))


if __name__=='__main__':main()
