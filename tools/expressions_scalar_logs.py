"""Frontend-generated natural-logarithm and compound-kernel cases.

Pinned Emmy CUDA rendering cannot emit log2/log10/exp2. Those operations have
independent typed-IR probes in generate_scalar_log_cases.py.
"""
import numpy as np


def log_softmax(x):
    x=x.astype(np.float64);m=x.max(-1,keepdims=True)
    return [(x-m-np.log(np.exp(x-m).sum(-1,keepdims=True))).astype(np.float32)]


def normalized_positive(x):
    positive=x.astype(np.float64)**2+.125
    return [(positive/positive.sum(-1,keepdims=True)).astype(np.float32)]


OPEN=[
    dict(id='p_log_257',code='torch.log(torch.abs(torch.randn(257))+.125)',
         ref=lambda x:[np.log(np.abs(x.astype(np.float64))+.125).astype(np.float32)]),
    dict(id='f_log_softmax_7_129',code='torch.log(torch.softmax(torch.randn(7,129),dim=-1))',ref=log_softmax),
    dict(id='f_log_sum_5_65',code='x=torch.randn(5,65); torch.log(x*x+.25).sum(-1)',
         ref=lambda x:[np.log(x.astype(np.float64)**2+.25).sum(-1).astype(np.float32)],knobs='WORK=t128,REDUCE=coop'),
    dict(id='f_positive_normalization_7_65',code='x=torch.randn(7,65); torch.softmax(torch.log(x*x+.125),dim=-1)',ref=normalized_positive),
]
for spec in OPEN:spec.update(rtol=1e-4,atol=2e-5)
