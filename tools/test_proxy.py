"""Tests for paired acquisition, disjoint pools, and group/seed metric weights."""
import copy
import math
import unittest
from prepare_proxy import split_group
from evaluate_proxy import random_choose
from score_proxy import aggregate, audit

CONFIG={'budgets':[0,1,3,7,15,31],'alc_weights':[.1,.2,.2,.2,.2,.1],
        'seeds':[42,43,44],'policies':['random','adaptive']}


class ProtocolTests(unittest.TestCase):
    def test_variable_complete_groups_split_without_truncation(self):
        for n in [80,95,160,501,1000]:
            g={'source':'s','subject_id':'u','items':[{'item_id':str(i)} for i in range(n)]}
            for seed in CONFIG['seeds']:
                a,e=split_group(g,seed)
                self.assertEqual((len(a),len(e)),((n+1)//2,n//2))
                self.assertEqual({r['item_id'] for r in a}|{r['item_id'] for r in e},{str(i) for i in range(n)})
                self.assertFalse({r['item_id'] for r in a}&{r['item_id'] for r in e})
                self.assertEqual((a,e),split_group(g,seed))

    def test_random_stream_exact_budget_and_no_future_inputs(self):
        for size in [40,80,250,500]:
            selected=[]
            for position in range(size):
                c={'labels_remaining':31-len(selected),'items_remaining':size-position,'subject_id':'s'}
                decision=random_choose([{'s':'s'},{'i':str(position)}],c)
                self.assertIsInstance(decision,bool)
                if decision:selected.append(position)
            self.assertEqual(len(selected),31)

    def test_equal_groups_and_equal_seeds(self):
        rows=[]
        for seed in CONFIG['seeds']:
            for policy in CONFIG['policies']:
                for source,n,y in [('a',1,1),('b',3,0)]:
                    for i in range(n):
                        rows.append(dict(source=source,subject_id='s',item_id=str(i),seed=seed,policy=policy,y=y,
                                         asrp=[0.]*6,empirical_mean=[.5]*6))
        result,_=aggregate(rows,CONFIG)
        self.assertAlmostEqual(result['combinations']['asrp_random']['alc'],.5)
        self.assertAlmostEqual(result['combinations']['empirical_mean_adaptive']['alc'],.25)
        self.assertEqual(result['acquisition_effect']['asrp']['adaptive_minus_random_alc'],0)
        with self.assertRaises(ValueError):aggregate(rows+[rows[0]],CONFIG)

    def test_official_log_budget_alc(self):
        b=[.25,.3,.22,.20,.18,.17];x=[math.log2(1+n) for n in CONFIG['budgets']]
        integral=sum((x[i+1]-x[i])*(b[i]+b[i+1])/2 for i in range(5))/(x[-1]-x[0])
        self.assertAlmostEqual(integral,sum(w*p for w,p in zip(CONFIG['alc_weights'],b)))

    def test_adaptive_policy_has_real_discretion_at_80_candidates(self):
        import sys
        from pathlib import Path
        import numpy as np
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/asrp'))
        from adaptive_gain_policy import AdaptiveGainPolicy
        called=[]
        def gram(items):
            called.append(len(items))
            x=np.asarray([[math.sin(int(i['item_content'])),math.cos(int(i['item_content']))] for i in items])
            return 2*(x@x.T)
        policy=AdaptiveGainPolicy(gram,power=4,warmup=16)
        support=[];decisions=[]
        for i in range(80):
            item={'item_content':str(i),'item_features':'','interactors':'','benchmark_id':'b'}
            pair=[{'normalized_name':'s'},item]
            context={'benchmark_id':'b','subject_id':'s','labels_remaining':31-len(support),
                     'labels_acquired':len(support),'max_labels':31,'items_remaining':80-i}
            chosen=policy.choose(pair,support,context);decisions.append(chosen)
            if chosen:support.append([pair,i%2])
        self.assertEqual(decisions[:16],[False]*16)
        self.assertEqual(len(support),31)
        self.assertGreater(len(called),0)
        self.assertIn(False,decisions[16:])


if __name__=='__main__':unittest.main()
