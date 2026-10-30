"""Public approximation of the documented PAIEC formative sampling protocol.

The private coordinator, its RNG and tie-breaking priorities are not public.
Local choices are explicit and deterministic; this is not organizer sampler code.
"""
from collections import Counter
import hashlib
import json

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (.1, .2, .2, .2, .2, .1)


def rank(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def balanced_draw(groups, seed, limit=1000):
    """Prefer less-represented benchmarks, then subjects; randomize ties.

    Counts reset within each draw. Never truncate a group to make it fit.
    Break benchmark ties before choosing a subject so benchmarks with many
    eligible subjects do not receive more initial probability. This hierarchy
    is our explicit interpretation of the public balancing description.
    """
    remaining = {g['group_id']: g for g in groups}
    benchmarks, subjects = Counter(), Counter()
    chosen, used = [], 0
    while True:
        feasible = [g for g in remaining.values() if g['n_items'] <= limit-used]
        if not feasible:
            break
        candidates={g['benchmark_alias'] for g in feasible}
        benchmark=min(candidates,key=lambda b:(benchmarks[b],rank(seed,'benchmark',len(chosen),b)))
        g = min((g for g in feasible if g['benchmark_alias']==benchmark),
                key=lambda g:(subjects[g['subject_alias']],rank(seed,'subject',len(chosen),g['group_id'])))
        chosen.append(g['group_id'])
        used += g['n_items']
        benchmarks[g['benchmark_alias']] += 1
        subjects[g['subject_alias']] += 1
        del remaining[g['group_id']]
    return {'seed': seed, 'group_ids': chosen, 'n_source_pairs': used}


def fixed_split(group, snapshot_seed):
    """All runs of an item stay together; the snapshot seed never uses draw ID."""
    ordered = sorted(group['items'], key=lambda i: rank(snapshot_seed, 'split', group['group_id'], i['item_id']))
    midpoint = (len(ordered)+1)//2
    acquisition, evaluation = ordered[:midpoint], ordered[midpoint:]
    acquisition.sort(key=lambda i: rank(snapshot_seed, 'stream', group['group_id'], i['item_id']))
    return acquisition, evaluation


def acquired_response(group, item, snapshot_seed):
    """Local deterministic selection of one recorded response, never an average."""
    return min(item['responses'], key=lambda r: rank(snapshot_seed, 'reveal', group['group_id'], r['response_id']))


def input_pair(group, item, response):
    return [group['subject'], {'item_content': item['content'], 'item_features': item['item_features'],
            'interactors': response['interactors'], 'benchmark_id': group['benchmark_alias']}]


def random_choose(item_input, context):
    # Same default rule as the public starting-kit streaming client.
    raw = json.dumps([context, item_input], sort_keys=True, separators=(',', ':'))
    uniform = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], 'big') / 2**64
    return uniform < min(1., context['labels_remaining']/context['items_remaining'])


def trajectory(groups, snapshot_seed, choose, checkpoint):
    """Checkpoint barriers and global evidence, with sequential group order.

    Only already acquired responses are passed to callbacks. Evaluation targets
    are not exposed to acquisition. All groups finish a cap before evaluation.
    """
    pools = [fixed_split(g, snapshot_seed)[0] for g in groups]
    cursors, counts = [0]*len(groups), [0]*len(groups)
    evidence, traces = [], []
    for budget in BUDGETS:
        for j, (g, pool) in enumerate(zip(groups, pools)):
            while counts[j] < budget and cursors[j] < len(pool):
                item = pool[cursors[j]]
                response = acquired_response(g, item, snapshot_seed)
                x = input_pair(g, item, response)
                context = {'subject_id': g['subject_alias'], 'benchmark_id': g['benchmark_alias'],
                           'labels_acquired': counts[j], 'labels_remaining': 31-counts[j],
                           'max_labels': 31, 'items_remaining': len(pool)-cursors[j]}
                yes = choose(x, evidence.copy(), context.copy())
                if type(yes) is not bool:
                    raise ValueError('Acquisition must return a native bool')
                traces.append({'group_id': g['group_id'], 'budget': budget, 'position': cursors[j],
                               'item_id': item['item_id'], 'acquired': yes, 'evidence_before': len(evidence),
                               'response_id': response['response_id'] if yes else None,
                               'label': response['label'] if yes else None})
                cursors[j] += 1
                if yes:
                    evidence.append([x, response['label']])
                    counts[j] += 1
        checkpoint(budget, evidence.copy(), counts.copy())
    return traces, counts
