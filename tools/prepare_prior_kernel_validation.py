"""Prepare unseen public groups for the locked no-Rasch candidate validation."""
from collections import defaultdict,Counter
import json
from pathlib import Path

import duckdb
from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from draw_protocol import rank,balanced_draw,fixed_split,acquired_response
from prepare_proxy import ROOT,FIELDS,sha,write_json


def main():
    protocol=json.loads((ROOT/'experiments/asrp_next/protocol.json').read_text())
    public=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    config=dict(public['config'])
    excluded=set(protocol['validation_excluded_group_ids'])
    eligible=[g for g in public['eligible_groups'] if g['group_id'] not in excluded]
    small=[g for g in eligible if 80<=g['n_items']<=protocol['small_pool']['max_group_items']]
    counts=Counter(g['benchmark_alias'] for g in small)
    pool=sorted([b for b,n in counts.items() if n>=2],key=lambda b:rank(protocol['small_pool']['pool_selection_seed'],b))[:7]
    assert len(pool)==7
    small=[g for g in small if g['benchmark_alias'] in pool]
    draws=[]
    for cohort,seeds,inventory in [('broad',protocol['validation_broad_seeds'],eligible),('small_pool',protocol['validation_small_pool_seeds'],small)]:
        for seed in seeds:draws.append({'draw_id':len(draws),'cohort':cohort,**balanced_draw(inventory,seed,1000)})
    selected={gid for d in draws for gid in d['group_ids']};assert not selected&excluded
    wanted=defaultdict(list)
    for g in eligible:
        if g['group_id'] in selected:wanted[g['source']].append(g)
    config['draw_seeds']=[d['seed'] for d in draws]
    config['validation_protocol']=protocol;config['small_pool_benchmark_aliases']=pool
    con=duckdb.connect(config={'threads':2,'memory_limit':'4GB'})
    def query(sql,args):
        r=con.execute(sql,args);return [dict(zip([c[0] for c in r.description],row)) for row in r.fetchall()]
    files={};splits=[];destination=ROOT/'data/asrp_next_validation.jsonl'
    with destination.open('w') as stream:
        for source,gg in sorted(wanted.items()):
            paths={}
            for name in ['subjects','items','responses']:
                filename=f'{source}/formatted_tables/{name}.parquet'
                paths[name]=hf_hub_download(config['dataset'],filename,repo_type='dataset',revision=config['revision'],local_files_only=True)
                assert sha(paths[name])==public['files'][filename]['sha256']
                files[filename]=public['files'][filename]
            subjects={r['subject_id']:{k:str(r[k] or '') for k in FIELDS} for r in query('SELECT * FROM read_parquet(?)',[paths['subjects']])}
            ids=set();responses={}
            for g in gg:
                rr=query('''SELECT item_id,response_id,trial,interactors,response FROM read_parquet(?)
                    WHERE subject_id=? AND benchmark_id=? AND response IN (0,1) ORDER BY item_id,trial,response_id''',[paths['responses'],g['subject_id'],g['benchmark_id']])
                mapping=defaultdict(list)
                for r in rr:mapping[r['item_id']].append({'response_id':r['response_id'],'trial':r['trial'],'label':int(r['response']),'interactors':str(r['interactors'] or '')})
                assert len(mapping)==g['n_items'] and sum(map(len,mapping.values()))==g['n_responses']
                responses[g['group_id']]=mapping;ids.update(mapping)
            items={};selection=pa.array(sorted(ids))
            for batch in pq.ParquetFile(paths['items']).iter_batches(batch_size=128,columns=['item_id','content','item_features']):
                for r in batch.filter(pc.is_in(batch.column('item_id'),value_set=selection)).to_pylist():items[r['item_id']]=r
            assert set(items)==ids
            for g in sorted(gg,key=lambda g:g['group_id']):
                subject=subjects[g['subject_id']]
                assert g['subject_alias']=='subject_'+rank('subject',g['subject_id'],subject)[:16]
                group=dict(g,subject=subject,items=[{'item_id':iid,'content':str(items[iid]['content'] or ''),'item_features':str(items[iid]['item_features'] or ''),'responses':rr} for iid,rr in sorted(responses[g['group_id']].items())])
                stream.write(json.dumps(group,ensure_ascii=False,separators=(',',':'))+'\n')
                acquisition,evaluation=fixed_split(group,config['snapshot_seed'])
                splits.append(dict(g,acquisition=[{'item_id':i['item_id'],'response_id':acquired_response(group,i,config['snapshot_seed'])['response_id']} for i in acquisition],
                    evaluation=[{'item_id':i['item_id'],'response_ids':[r['response_id'] for r in i['responses']]} for i in evaluation]))
            print(f'Prepared validation source {source}: {len(gg)} groups',flush=True)
    con.close()
    write_json(ROOT/'experiments/asrp_next/validation_manifest.json',{'config':config,'files':files,'draws':draws,'groups':sorted(splits,key=lambda g:g['group_id']),
        'prepared_sha256':sha(destination),'protocol_sha256':sha(ROOT/'experiments/asrp_next/protocol.json'),'script_sha256':sha(Path(__file__))})
    print(f'Validation locked before model selection: {len(draws)} draws, {len(splits)} distinct groups, zero development-group overlap.',flush=True)


if __name__=='__main__':main()
