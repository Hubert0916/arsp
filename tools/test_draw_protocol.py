"""Scientific protocol invariants on synthetic data; no model/data downloads."""
import unittest
from draw_protocol import balanced_draw, fixed_split, trajectory, random_choose, BUDGETS


def group(name,benchmark,subject,n=81):
    return {'group_id':name,'benchmark_alias':benchmark,'subject_alias':subject,
        'subject':{'normalized_name':subject},'n_items':n,'items':[
        {'item_id':f'{name}-{i}','content':str(i),'item_features':'','responses':[
         {'response_id':f'{name}-{i}-{r}','label':r,'interactors':''} for r in (0,1)]} for i in range(n)]}


class ProtocolTest(unittest.TestCase):
    def test_budget_complete_groups_and_balancing(self):
        gs=[group(f'g{b}{s}',f'b{b}',f's{s}',81) for b in range(3) for s in range(3)]
        a=balanced_draw(gs,71,330);b=balanced_draw(list(reversed(gs)),71,330)
        self.assertEqual(a,b)
        self.assertEqual(a['n_source_pairs'],324)
        selected=[next(g for g in gs if g['group_id']==k) for k in a['group_ids']]
        self.assertEqual(len({g['benchmark_alias'] for g in selected[:3]}),3)
        self.assertEqual(len({g['subject_alias'] for g in selected[:3]}),3)
        self.assertEqual(len(set(a['group_ids'])),4)
        self.assertEqual(balanced_draw([group('too_big','b','s',1001)],71)['group_ids'],[])

    def test_more_subjects_do_not_bias_initial_benchmark_tie(self):
        base=[group('a','b0','s0',80),group('b','b1','s1',80)]
        expanded=base+[group(f'extra{i}','b1',f's{i+2}',80) for i in range(20)]
        lookup={g['group_id']:g['benchmark_alias'] for g in expanded}
        for seed in range(20):
            first=balanced_draw(base,seed,80)['group_ids'][0]
            second=balanced_draw(expanded,seed,80)['group_ids'][0]
            self.assertEqual(lookup[first],lookup[second])

    def test_fixed_item_partition_retains_all_response_runs(self):
        g=group('g','b','s')
        a,e=fixed_split(g,1)
        self.assertEqual((len(a),len(e)),(41,40))
        self.assertFalse({r['item_id'] for r in a}&{r['item_id'] for r in e})
        self.assertEqual(sum(len(i['responses']) for i in e),80)
        self.assertEqual((a,e),fixed_split(g,1))
        self.assertNotEqual((a,e),fixed_split(g,2))

    def test_global_evidence_checkpoint_barrier_and_no_eval_leak(self):
        gs=[group('a','b','s1'),group('c','b','s2')]
        pools={g['subject_alias']:{i['content'] for i in fixed_split(g,9)[0]} for g in gs}
        stages=[];sizes=[]
        def choose(x,evidence,context):
            self.assertIn(x[1]['item_content'],pools[context['subject_id']])
            self.assertEqual(context['labels_remaining'],31-context['labels_acquired'])
            sizes.append(len(evidence))
            return True
        def checkpoint(b,evidence,counts):
            self.assertEqual(counts,[b,b]);self.assertEqual(len(evidence),2*b)
            self.assertEqual(set(r[1] for r in evidence)-{0,1},set())
            stages.append(b)
        traces,counts=trajectory(gs,9,choose,checkpoint)
        self.assertEqual(stages,list(BUDGETS));self.assertEqual(counts,[31,31])
        self.assertEqual(sizes,list(range(62)))
        self.assertEqual(len({(r['group_id'],r['item_id']) for r in traces}),62)

    def test_random_reaches_cap_without_repeat(self):
        gs=[group('a','b','s')]
        traces,counts=trajectory(gs,8,lambda x,e,c:random_choose(x,c),lambda *args:None)
        self.assertEqual(counts,[31])
        self.assertEqual(sum(r['acquired'] for r in traces),31)


if __name__=='__main__':unittest.main()
