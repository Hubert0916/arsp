"""Prepare complete public groups and fixed pools for repeated formative draws."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import duckdb
from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prepare_proxy import ROOT, FIELDS, sha, write_json
from draw_protocol import rank, balanced_draw, fixed_split, acquired_response


def prepare(config, cache_only):
    old = json.loads((ROOT/'experiments/proxy_manifest.json').read_text())
    con = duckdb.connect(config={'threads': 2, 'memory_limit': '3GB'})
    files, tables, inventory, eligible = {}, {}, [], []
    def query(sql, args):
        q=con.execute(sql,args)
        return [dict(zip([x[0] for x in q.description],r)) for r in q.fetchall()]
    for source in config['sources']:
        paths={}
        for name in ('benchmarks','subjects','items','responses'):
            filename=f'{source}/formatted_tables/{name}.parquet'
            paths[name]=hf_hub_download(config['dataset'],filename,repo_type='dataset',revision=config['revision'],local_files_only=cache_only)
            checksum=sha(paths[name])
            if checksum!=old['files'][filename]['sha256']:
                raise ValueError('Pinned table changed: '+filename)
            files[filename]=old['files'][filename]
        tables[source]=paths
        subjects={r['subject_id']:{k:str(r[k] or '') for k in FIELDS} for r in query('SELECT * FROM read_parquet(?)',[paths['subjects']])}
        binary={r['benchmark_id'] for r in query('SELECT benchmark_id, response_type FROM read_parquet(?)',[paths['benchmarks']]) if r['response_type']=='binary'}
        counts=query('''SELECT benchmark_id, subject_id, count(DISTINCT item_id) AS n_items,
                       count(*) AS n_responses FROM read_parquet(?) WHERE response IN (0,1)
                       GROUP BY benchmark_id,subject_id''',[paths['responses']])
        valid=[]
        for r in counts:
            if r['benchmark_id'] not in binary or not 80<=r['n_items']<=config['draw_limit']:
                continue
            subject=subjects[r['subject_id']]
            g={**r,'source':source,'subject':subject,
               'group_id':rank(source,r['benchmark_id'],r['subject_id'])[:20],
               'benchmark_alias':'benchmark_'+rank('benchmark',source,r['benchmark_id'])[:16],
               'subject_alias':'subject_'+rank('subject',r['subject_id'],subject)[:16]}
            valid.append(g)
        # Keep upstream identities even if missing visible attributes make two
        # different subjects indistinguishable to the submitted predictor.
        keys=[(g['benchmark_alias'],json.dumps(g['subject'],sort_keys=True)) for g in valid]
        eligible.extend(valid)
        inventory.append({'source':source,'eligible_groups':len(valid),
                          'duplicate_visible_identities':len(keys)-len(set(keys)),
                          'groups_below_80':sum(r['n_items']<80 for r in counts),
                          'groups_above_draw_limit':sum(r['n_items']>config['draw_limit'] for r in counts)})
        print(f"Indexed {len(inventory)}/{len(config['sources'])}: {source}, {len(valid)} eligible groups",flush=True)
    # DuckDB's parallel GROUP BY output order is not stable. Canonicalize the
    # inventory before writing provenance; sampling already uses seeded ranks.
    eligible.sort(key=lambda g:(g['source'],g['benchmark_id'],g['subject_id']))
    draws=[{'draw_id':j,**balanced_draw(eligible,seed,config['draw_limit'])} for j,seed in enumerate(config['draw_seeds'])]
    selected={gid for d in draws for gid in d['group_ids']}
    wanted=defaultdict(list)
    for g in eligible:
        if g['group_id'] in selected:wanted[g['source']].append(g)
    groups=[]
    for source,chosen in sorted(wanted.items()):
        paths=tables[source];all_ids=set();responses={}
        for g in chosen:
            rows=query('''SELECT item_id,response_id,trial,interactors,response FROM read_parquet(?)
                WHERE subject_id=? AND benchmark_id=? AND response IN (0,1) ORDER BY item_id,trial,response_id''',
                [paths['responses'],g['subject_id'],g['benchmark_id']])
            mapping=defaultdict(list)
            for r in rows:
                mapping[r['item_id']].append({'response_id':r['response_id'],'trial':r['trial'],
                    'label':int(r['response']),'interactors':str(r['interactors'] or '')})
            assert len(mapping)==g['n_items'] and sum(map(len,mapping.values()))==g['n_responses']
            responses[g['group_id']]=mapping;all_ids.update(mapping)
        items={}
        selection=pa.array(sorted(all_ids))
        for batch in pq.ParquetFile(paths['items']).iter_batches(batch_size=128,columns=['item_id','content','item_features']):
            for r in batch.filter(pc.is_in(batch.column('item_id'),value_set=selection)).to_pylist():
                if r['item_id'] in items:raise ValueError('Duplicate item')
                items[r['item_id']]=r
        assert set(items)==all_ids
        for g in chosen:
            g={**g,'items':[{'item_id':iid,'content':str(items[iid]['content'] or ''),
                 'item_features':str(items[iid]['item_features'] or ''),'responses':rr}
                 for iid,rr in sorted(responses[g['group_id']].items())]}
            groups.append(g)
        print(f'Loaded {source}: {len(chosen)} groups, {len(items)} distinct source items',flush=True)
    con.close()
    groups.sort(key=lambda g:g['group_id'])
    splits=[]
    for g in groups:
        pool,targets=fixed_split(g,config['snapshot_seed'])
        splits.append({**{k:g[k] for k in ('group_id','source','benchmark_id','subject_id','benchmark_alias','subject_alias','n_items','n_responses')},
           'acquisition':[{'item_id':i['item_id'],'response_id':acquired_response(g,i,config['snapshot_seed'])['response_id']} for i in pool],
           'evaluation':[{'item_id':i['item_id'],'response_ids':[r['response_id'] for r in i['responses']]} for i in targets]})
    return groups,{'config':config,'files':files,'inventory':inventory,
                   'eligible_groups':[{k:v for k,v in g.items() if k!='subject'} for g in eligible],
                   'draws':draws,'groups':splits}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-only',action='store_true')
    p.add_argument('--write-manifest',action='store_true')
    args=p.parse_args()
    config=json.loads((ROOT/'experiments/formative_config.json').read_text())
    groups,manifest=prepare(config,args.cache_only)
    payload=''.join(json.dumps(g,ensure_ascii=False,separators=(',',':'))+'\n' for g in groups)
    import hashlib
    manifest['prepared_sha256']=hashlib.sha256(payload.encode()).hexdigest()
    path=ROOT/'experiments/formative_manifest.json'
    if args.write_manifest:write_json(path,manifest)
    else:
        expected=json.loads(path.read_text())
        differences=[k for k in set(manifest)|set(expected) if manifest.get(k)!=expected.get(k)]
        if differences:raise ValueError('Prepared groups/draws differ from release: '+', '.join(sorted(differences)))
    (ROOT/'data').mkdir(parents=True,exist_ok=True)
    (ROOT/'data/formative.jsonl').write_text(payload)
    print(f'Prepared {len(groups)} distinct groups for {len(manifest["draws"])} draws.',flush=True)


if __name__=='__main__':main()
