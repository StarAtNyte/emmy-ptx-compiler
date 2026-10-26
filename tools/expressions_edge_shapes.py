"""Tiny, odd and degenerate shapes in every category.

Idle lanes, single-element rows and one-wide contractions stress guards,
strided-loop tails and reduction identities. Same generator and tolerances
as the open set; float64 NumPy references.
"""
import numpy as np


OPEN = []
REDUCE = 'WORK=t128,REDUCE=coop'


def case(name, code, ref, *, rtol=1e-3, atol=1e-4, knobs=''):
    OPEN.append(dict(id=name, code=code, ref=ref, knobs=knobs, rtol=rtol, atol=atol))


def f64(x):
    return x.astype(np.float64)


def out(x):
    return x.astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


for n in (1, 2, 3, 5, 31, 33, 65, 257):
    case(f'eg_p_scale_{n}', f'torch.randn({n})*2.0+1.0', lambda x: [x*np.float32(2)+np.float32(1)], rtol=1e-6, atol=1e-6)
for n in (2, 3, 17, 129, 4097):
    case(f'eg_r_sum_{n}', f'torch.randn({n}).sum()', lambda x: [out(f64(x).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE)
    case(f'eg_r_max_{n}', f'torch.randn({n}).amax()', lambda x: [x.max()], rtol=0, atol=0, knobs=REDUCE)
for rows, cols in ((1, 1), (1, 7), (7, 1), (3, 2), (2, 3), (5, 33), (33, 5), (1, 1000)):
    case(f'eg_r_rows_{rows}_{cols}', f'torch.randn({rows},{cols}).sum(-1)', lambda x: [out(f64(x).sum(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE)
    case(f'eg_f_softmax_{rows}_{cols}', f'torch.softmax(torch.randn({rows},{cols}),dim=-1)', lambda x: [out(softmax(f64(x)))], atol=1e-5)
    case(f'eg_f_rms_{rows}_{cols}', f'x=torch.randn({rows},{cols}); x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)',
         lambda x: [out(f64(x)/np.sqrt((f64(x)**2).mean(-1, keepdims=True)+1e-5))])
for m, k, n in ((1, 1, 1), (1, 7, 1), (5, 1, 3), (2, 3, 4), (1, 128, 64), (64, 128, 1), (3, 300, 5)):
    case(f'eg_m_{m}_{k}_{n}', f'torch.randn({m},{k}) @ torch.randn({k},{n})', lambda a, b: [out(f64(a)@f64(b))], atol=1e-3)
for seq, depth in ((1, 4), (2, 2), (3, 5), (5, 3), (7, 16), (16, 1), (33, 8), (9, 64)):
    def attention(q, kt, v, depth=depth):
        return [out(softmax(f64(q)@f64(kt)/np.sqrt(depth))@f64(v))]
    case(f'eg_a_{seq}_{depth}',
         f'q=torch.randn({seq},{depth}); kt=torch.randn({depth},{seq}); v=torch.randn({seq},{depth}); torch.softmax(q@kt/{float(np.sqrt(depth))},dim=-1)@v',
         attention)
