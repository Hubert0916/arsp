"""Audit native component predictions saved by the paired ASRP evaluator."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

from prepare_proxy import ROOT, sha, write_json

VARIANTS = ('full', 'no_kernel', 'no_topic')


def read_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def identity(row):
    return tuple(row[k] for k in ('source','subject_id','seed','policy','item_id'))


def score(directory, compare=None):
    manifest = json.loads((ROOT/'experiments/proxy_manifest.json').read_text())
    config = manifest['config']
    protocol = json.loads((ROOT/'experiments/ablation_config.json').read_text())
    run = json.loads((directory/'run.json').read_text())
    if run['inference_manifest_sha256'] != sha(ROOT/'artifacts/manifest.sha256.json'):
        raise ValueError('Component run belongs to different frozen inference files')
    if run['split_manifest_sha256'] != sha(ROOT/'experiments/proxy_manifest.json') or run['config'] != config:
        raise ValueError('Ablation run differs from the published protocol')
    if list(protocol['variants']) != list(VARIANTS):
        raise ValueError('Unexpected ablation definitions')
    for name,expected in run['files'].items():
        if sha(directory/name) != expected: raise ValueError('Run output checksum mismatch: '+name)
    reference = {identity(r):r for r in read_rows(directory/'predictions.jsonl') if r['policy']=='adaptive'}
    records = read_rows(directory/'component_predictions.jsonl.gz')
    cells,seen = defaultdict(list),set()
    for row in records:
        key=identity(row)
        if key in seen or key not in reference: raise ValueError('Duplicate or unexpected component row')
        seen.add(key); original=reference[key]
        if row['y'] != original['y'] or row['response_id'] != original['response_id']:
            raise ValueError('Component evaluation outcome changed')
        if row['full'] != original['asrp']: raise ValueError('Full component prediction differs from ASRP')
        for variant in VARIANTS:
            if len(row[variant]) != 6 or any(not math.isfinite(p) or not 0<=p<=1 for p in row[variant]):
                raise ValueError('Invalid component probability array')
        if row['no_kernel'][:2] != row['full'][:2]:
            raise ValueError('Removing a zero-weight kernel changed cold-start predictions')
        cells[(row['source'],row['subject_id'],row['seed'])].append(row)
    expected={(g['source'],g['subject_id'],s) for g in manifest['groups'] for s in config['seeds']}
    if seen != set(reference) or set(cells) != expected: raise ValueError('Incomplete component coverage')
    by_group=[]
    for (source,subject,seed),rows in sorted(cells.items()):
        for variant in VARIANTS:
            curve=[math.fsum((r[variant][j]-r['y'])**2 for r in rows)/len(rows) for j in range(6)]
            by_group.append({'source':source,'subject_id':subject,'seed':seed,'variant':variant,
                             'n_rows':len(rows),'brier':curve,
                             'alc':math.fsum(w*b for w,b in zip(config['alc_weights'],curve))})
    variants={}
    for variant in VARIANTS:
        per_seed=[]
        for seed in config['seeds']:
            rows=[r for r in by_group if r['variant']==variant and r['seed']==seed]
            per_seed.append({'seed':seed,'alc':statistics.mean(r['alc'] for r in rows),
                             'brier':[statistics.mean(r['brier'][j] for r in rows) for j in range(6)]})
        variants[variant]={'alc':statistics.mean(r['alc'] for r in per_seed),
                           'brier':[statistics.mean(r['brier'][j] for r in per_seed) for j in range(6)],
                           'alc_seed_sd':statistics.stdev(r['alc'] for r in per_seed),'per_seed':per_seed}
    for value in variants.values(): value['delta_alc_vs_full']=value['alc']-variants['full']['alc']
    result={'protocol':protocol,'n_groups':len(manifest['groups']),'n_prediction_records':len(records),
            'variants':variants,'native_component_file':'component_predictions.jsonl.gz'}
    write_json(directory/'ablation_summary.json',result)
    write_json(directory/'ablation_by_source.json',by_group)
    if compare is not None:
        reference_rows=read_rows(compare/'component_predictions.jsonl.gz')
        if len(reference_rows)!=len(records): raise ValueError('Component reference coverage differs')
        for row,other in zip(records,reference_rows):
            if identity(row)!=identity(other) or row['y']!=other['y'] or row['response_id']!=other['response_id']:
                raise ValueError('Component reference identities differ')
            for variant in VARIANTS:
                if any(abs(a-b)>1e-7 for a,b in zip(row[variant],other[variant])):
                    raise ValueError('Component predictions differ by more than 1e-7')
        print('All component predictions match the published values within 1e-7.')
    print(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=ROOT/'results/proxy')
    parser.add_argument('--compare',type=Path)
    args=parser.parse_args()
    score(args.directory,args.compare)


if __name__=='__main__':main()
