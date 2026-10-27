"""Streaming experimental design with a nonlinear, support-adaptive kernel."""
import hashlib
import json

import numpy as np

from base_features import feature_key, subject_key


class AdaptiveGainPolicy:
    def __init__(self, gram, key=feature_key, power=1., warmup=0, window=64):
        self.gram,self.key,self.power,self.warmup,self.window=gram,key,power,warmup,window
        self.seen,self.counts={},{}

    def choose(self,input,evidence,context):
        subject,item=input
        benchmark=item.get('benchmark_id') or context['benchmark_id']
        seen=self.seen.setdefault(benchmark,[])
        past=list({self.key(i):i for i in seen[-self.window:]}.values())
        seen.append(item.copy())
        current=subject_key(subject)
        pair=benchmark,current
        count,initial_total=self.counts.get(pair,(0,context['items_remaining']))
        self.counts[pair]=(count+1,initial_total)
        remaining,total=context['labels_remaining'],context['items_remaining']
        if remaining<=0: return False
        if total<=remaining: return True
        if count<min(self.warmup,max(0,initial_total-context['max_labels'])):
            return False
        raw=json.dumps([context,input],sort_keys=True,separators=(',',':'))
        u=int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8],'big')/2**64
        if not past or self.power==0: return bool(u<remaining/total)
        records=[(s,i,y) for (s,i),y in evidence if i.get('benchmark_id')==benchmark]
        who=[subject_key(s) for s,i,y in records]
        n=len(records)
        items=[*[i for s,i,y in records],*past,item]
        rawk=self.gram(items)/2
        # Bandwidth uses only revealed/current/past inputs, never the future pool.
        reference=rawk[n:-1,n:-1]
        distances=np.maximum(1-reference[np.triu_indices(len(reference),1)],0)
        positive=distances[distances>1e-8]
        width=4*max(float(np.median(positive)) if len(positive) else .1,1e-5)
        k=np.exp(np.minimum(rawk-1,0)/width)
        same=np.asarray([[a==b for a in who] for b in who],float).reshape(n,n)
        target=np.asarray([w==current for w in who],float)
        covariance=k[n:,n:]+.25
        if n:
            train=k[:n,:n]+.25*same
            cross=k[n:,:n]+.25*target[None,:]
            covariance-=cross @ np.linalg.solve(train+np.eye(n),cross.T)
        variance=np.maximum(np.diag(covariance),0)
        gain=np.maximum((covariance[:-1]**2).mean(0)/(1+variance),1e-12)
        relative=gain/max(float(gain[:-1].mean()),1e-12)
        utility=np.minimum(relative,20)**self.power
        ratio=float(utility[-1]/max(float(utility[:-1].mean()),1e-12))
        return bool(u<min(1,remaining/total*np.clip(ratio,.05,20)))
