"""Runnable corpus from the frozen-compiler evaluation.

Original attempts and the callable-boundary correction are preserved under
experiments/. Two upstream-untraceable cases are excluded. Compiler code was
unchanged throughout the evaluation.
"""
import numpy as np

_rng=np.random.default_rng(20261024)
OPEN=[]


def add(name,code,ref,knobs='',rtol=1e-3,atol=1e-4):
    OPEN.append(dict(id=name,code=code,ref=ref,knobs=knobs,rtol=rtol,atol=atol))


sizes=_rng.choice([43,89,173,271,541,1031],size=3,replace=False)
add('p_holdout_mixed',f'x=torch.randn({sizes[0]}); y=torch.randn({sizes[0]}); torch.tanh(x)*torch.sigmoid(y)+torch.relu(x-y)',
    lambda x,y:[np.tanh(x)/(1+np.exp(-y))+np.maximum(x-y,0)])
add('p_holdout_sqrt',f'x=torch.randn({sizes[1]}); torch.sqrt(x*x+0.125)',
    lambda x:[np.sqrt(x*x+np.float32(.125))])
add('p_holdout_minmax',f'x=torch.randn({sizes[2]}); y=torch.randn({sizes[2]}); torch.minimum(x,y)-torch.maximum(-x,y)',
    lambda x,y:[np.minimum(x,y)-np.maximum(-x,y)])
add('r_holdout_3d','torch.randn(2,5,137).sum(-1)',
    lambda x:[x.astype(np.float64).sum(-1).astype(np.float32)],'WORK=t128,REDUCE=coop')
for index in range(2):
    m,k,n=map(int,_rng.choice([19,29,43,53,67],size=3,replace=False))
    add(f'm_holdout_{index}',f'torch.randn({m},{k})@torch.randn({k},{n})',
        lambda a,b:[(a.astype(np.float64)@b.astype(np.float64)).astype(np.float32)],atol=1e-3)


def layernorm(x):
    y=x.astype(np.float64)-x.astype(np.float64).mean(-1,keepdims=True)
    return [(y/np.sqrt((y*y).mean(-1,keepdims=True)+1e-5)).astype(np.float32)]


add('f_holdout_layernorm_module','''
class Normalization(nn.Module):
    def forward(self,x):
        y=x-x.mean(-1,keepdim=True)
        return y/torch.sqrt((y*y).mean(-1,keepdim=True)+1e-5)
Normalization()(torch.randn(5,83))
''',layernorm)


def softmax(x):
    x=x.astype(np.float64)
    e=np.exp(x-x.max(-1,keepdims=True))
    return [(e/e.sum(-1,keepdims=True)).astype(np.float32)]


add('f_holdout_softmax','torch.softmax(torch.randn(9,73),dim=-1)',softmax,'WORK=t64,REDUCE=coop')
for index in range(2):
    queries,keys,depth,values=map(int,_rng.choice([7,11,17,23,31,41],size=4,replace=False))
    def attention(q,kt,v,scale=np.sqrt(depth)):
        scores=q.astype(np.float64)@kt.astype(np.float64)/scale
        e=np.exp(scores-scores.max(-1,keepdims=True))
        return [(e/e.sum(-1,keepdims=True)@v.astype(np.float64)).astype(np.float32)]
    add(f'a_holdout_{index}',
        f'q=torch.randn({queries},{depth}); kt=torch.randn({depth},{keys}); v=torch.randn({keys},{values}); torch.softmax(q@kt/{float(np.sqrt(depth))},dim=-1)@v',attention)
