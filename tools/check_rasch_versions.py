"""Compare the archived no-Rasch implementation with the controlled ablation.

Every released draw/policy/checkpoint is checked on the first and last distinct
evaluation inputs of each group. Evidence is replayed from the released traces.
This verifies code versions, not the contents of an unavailable platform ZIP.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gzip
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import evaluate_proxy as runtime
from evaluate_ablation import disabled
from draw_protocol import BUDGETS, fixed_split, input_pair
from prepare_proxy import ROOT, sha, write_json
from smoke_test import load_module

OLD = '2f85fdb'


def initialize(current, removed, config):
    import sys
    global V2
    runtime.init_worker(current, config)
    sys.path.insert(0, str(removed))
    V2 = load_module(Path(removed)/'model.py', 'archived_no_rasch')
    original = V2._item_embedding
    @lru_cache(maxsize=4096)
    def embedding(encoder, text):
        return original(encoder, {'item_content': text})
    V2._item_embedding = lambda encoder, item: embedding(encoder, item.get('item_content', ''))


def check(task):
    from threadpoolctl import threadpool_limits
    did, policy, checkpoints = task
    maximum, n = 0., 0
    with threadpool_limits(limits=2):
        for budget, evidence, targets in checkpoints:
            for x in targets:
                with disabled(runtime.MODEL, 'no_rasch'):
                    a = runtime.finite_probability(runtime.MODEL.predict(x, evidence))
                b = runtime.finite_probability(V2.predict(x, evidence))
                maximum = max(maximum, abs(a-b)); n += 1
                if abs(a-b)>1e-12:
                    raise ValueError(f'Archived implementation mismatch: {did}, {policy}, {budget}: {a}, {b}')
    return {'draw_id': did, 'policy': policy, 'n_predictions': n, 'max_error': maximum}


def semantic_code(text):
    """Ignore module documentation when comparing the shared kernel module."""
    tree = ast.parse(text)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'experiments/formative_manifest.json').read_text())
    if sha(ROOT/'data/formative.jsonl')!=manifest['prepared_sha256']:
        raise ValueError('Prepared input checksum mismatch')
    with (ROOT/'data/formative.jsonl').open() as f:
        groups={g['group_id']:g for g in map(json.loads,f)}
    queries={};observations={}
    for gid,g in groups.items():
        for item in g['items']:
            for response in item['responses']:
                # Only acquisition responses need storage; repeated evaluation
                # outcomes do not change which code inputs are selected.
                observations.setdefault((gid,item['item_id']),{})[response['response_id']]=response
        _,targets=fixed_split(g,manifest['config']['snapshot_seed'])
        distinct={}
        for item in targets:
            for response in item['responses']:
                x=input_pair(g,item,response)
                distinct.setdefault(json.dumps(x,sort_keys=True),x)
        xs=list(distinct.values())
        queries[gid]=[xs[0]] if len(xs)==1 else [xs[0],xs[-1]]
    with gzip.open(ROOT/'results/formative/acquisition.jsonl.gz','rt') as f:
        traces=[json.loads(line) for line in f]
    tasks=[]
    items={gid:{i['item_id']:i for i in g['items']} for gid,g in groups.items()}
    for t in traces:
        checkpoints=[]
        for budget in BUDGETS:
            evidence=[]
            for d in t['decisions']:
                if d['acquired'] and d['budget']<=budget:
                    gid=d['group_id'];item=items[gid][d['item_id']]
                    response=observations[(gid,d['item_id'])][d['response_id']]
                    assert response['label']==d['label']
                    evidence.append([input_pair(groups[gid],item,response),d['label']])
            checkpoints.append((budget,evidence,[x for gid in t['groups'] for x in queries[gid]]))
        tasks.append((t['draw_id'],t['policy'],checkpoints))
    del groups,items,observations
    frozen=json.loads((ROOT/'artifacts/manifest.sha256.json').read_text())
    with tempfile.TemporaryDirectory(prefix='asrp-version-audit-') as td:
        root=Path(td);current=root/'current';removed=root/'removed'
        for name,digest in frozen.items():
            if sha(ROOT/name)!=digest:raise ValueError('Current inference changed')
            prefix='src/asrp' if name.startswith('src/') else 'artifacts/asrp'
            target=current/Path(name).relative_to(prefix);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        tar=root/'removed.tar'
        with tar.open('wb') as f:
            subprocess.run(['git','archive',OLD,'src/asrp','artifacts/asrp','artifacts/manifest.sha256.json'],cwd=ROOT,stdout=f,check=True)
        with tarfile.open(tar) as f:f.extractall(root/'snapshot',filter='data')
        old_manifest=json.loads((root/'snapshot/artifacts/manifest.sha256.json').read_text())
        for name,digest in old_manifest.items():
            if sha(root/'snapshot'/name)!=digest:raise ValueError('Archived inference changed')
            prefix='src/asrp' if name.startswith('src/') else 'artifacts/asrp'
            target=removed/Path(name).relative_to(prefix);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/'snapshot'/name,target)
        shared=['labeling.py','adaptive_gain_policy.py','base_features.py','evidence_features.py',
                'project_core.py','project_policy.py','window_features.py','linear_gain_policy.py',
                'balanced_probe_policy.py']
        assert all(sha(current/n)==sha(removed/n) for n in shared)
        assert semantic_code((current/'kernel_probe.py').read_text())==semantic_code((removed/'kernel_probe.py').read_text())
        # Imports share the kernel implementation only after verifying semantic
        # identity. Each model loads its own encoder/configuration/artifacts.
        rows=[]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(current,removed,manifest['config'])) as pool:
            futures=[pool.submit(check,t) for t in tasks]
            for f in as_completed(futures):
                rows.append(f.result());print(f'Checked {len(rows)}/{len(tasks)} draw/policy trajectories',flush=True)
    rows.sort(key=lambda r:(r['draw_id'],r['policy']))
    write_json(ROOT/'results/rasch_audit/version_check.json',{
        'archived_commit':subprocess.check_output(['git','rev-parse',OLD],cwd=ROOT,text=True).strip(),
        'scope':'First and last distinct evaluation inputs in every group of all 30 draws, both policies, six budgets; not all evaluation inputs.',
        'shared_acquisition_files_identical':shared,'shared_kernel_ast_identical':True,
        'n_predictions':sum(r['n_predictions'] for r in rows),'max_error':max(r['max_error'] for r in rows),
        'formative_manifest_sha256':sha(ROOT/'experiments/formative_manifest.json'),
        'current_inference_manifest_sha256':sha(ROOT/'artifacts/manifest.sha256.json'),
        'script_sha256':sha(Path(__file__)),'trajectories':rows,
        'historical_platform_zip_hash_available':False})


if __name__=='__main__':main()
