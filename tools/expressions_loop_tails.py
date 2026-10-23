"""Prime-sized dot products exercise unrolled loops and scalar epilogues."""
import numpy as np

OPEN=[]
for queries,keys,depth,values in ((11,13,19,7),(17,23,37,11)):
    def reference(q,kt,v,scale=np.sqrt(depth)):
        scores=q.astype(np.float64)@kt.astype(np.float64)/scale
        exp=np.exp(scores-scores.max(-1,keepdims=True))
        return [(exp/exp.sum(-1,keepdims=True)@v.astype(np.float64)).astype(np.float32)]
    OPEN.append(dict(id=f'a_cross_{queries}_{keys}_{depth}_{values}',
        code=f'q=torch.randn({queries},{depth}); kt=torch.randn({depth},{keys}); v=torch.randn({keys},{values}); torch.softmax(q@kt/{float(np.sqrt(depth))},dim=-1)@v',
        ref=reference,rtol=1e-3,atol=1e-4))
