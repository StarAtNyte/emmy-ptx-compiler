"""Independent expression corpus for the organizers' problem generator.

Generate with the pinned Emmy release and sm_89, using challenge/tools/generate.py.
These deliberately vary shapes and op mixes instead of editing serialized IR.
"""
import numpy as np


OPEN = []


def case(name, code, ref, *, knobs='', rtol=1e-3, atol=1e-4):
    OPEN.append(dict(id=name, code=code, ref=ref, knobs=knobs, rtol=rtol, atol=atol))


for size in (1, 31, 127, 513, 3072):
    case(f'p_add_{size}', f'torch.randn({size}) + torch.randn({size})',
         lambda a, b: [a + b], rtol=0, atol=0)

case('p_sub_div', 'x=torch.randn(257); y=torch.randn(257); (x-y)/(y*y+0.5)',
     lambda x, y: [(x-y)/(y*y+np.float32(.5))])
case('p_exp', 'torch.exp(torch.randn(512))', lambda x: [np.exp(x)])
case('p_sigmoid', 'torch.sigmoid(torch.randn(1024))', lambda x: [1/(1+np.exp(-x))])
case('p_tanh', 'torch.tanh(torch.randn(128))', lambda x: [np.tanh(x)])

for size in (17, 257, 8193):
    case(f'r_sum_{size}', f'torch.randn({size}).sum()',
         lambda x: [x.astype(np.float64).sum().astype(np.float32)],
         knobs='WORK=t128,REDUCE=coop')

case('r_max_rows', 'torch.randn(7, 257).amax(-1)', lambda x: [x.max(-1)],
     knobs='WORK=t128,REDUCE=coop')
case('r_mean_rows', 'torch.randn(9, 96).mean(-1)',
     lambda x: [x.astype(np.float64).mean(-1).astype(np.float32)],
     knobs='WORK=t128,REDUCE=coop')

for rows, cols in ((3, 33), (5, 129), (8, 1024)):
    def softmax(x):
        x = x.astype(np.float64)
        e = np.exp(x - x.max(-1, keepdims=True))
        return [(e/e.sum(-1, keepdims=True)).astype(np.float32)]
    case(f'f_softmax_{rows}_{cols}', f'torch.softmax(torch.randn({rows},{cols}),dim=-1)', softmax)

case('f_rms_7_96', 'x=torch.randn(7,96); w=torch.randn(96); x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)*w',
     lambda x,w: [(x/np.sqrt((x.astype(np.float64)**2).mean(-1,keepdims=True)+1e-5)*w).astype(np.float32)])

for m,k,n in ((16,32,16),(33,17,25),(64,96,32)):
    case(f'm_mm_{m}_{k}_{n}', f'torch.randn({m},{k}) @ torch.randn({k},{n})',
         lambda a,b: [(a.astype(np.float64)@b.astype(np.float64)).astype(np.float32)], atol=1e-3)

for seq,depth in ((16,16),(32,16)):
    def attention(q,kt,v,scale=np.sqrt(depth)):
        scores = q.astype(np.float64)@kt.astype(np.float64)/scale
        e = np.exp(scores-scores.max(-1,keepdims=True))
        return [(e/e.sum(-1,keepdims=True)@v.astype(np.float64)).astype(np.float32)]
    case(f'a_attention_{seq}_{depth}',
         f'q=torch.randn({seq},{depth}); kt=torch.randn({depth},{seq}); v=torch.randn({seq},{depth}); torch.softmax(q@kt/{float(np.sqrt(depth))},dim=-1)@v',
         attention)
