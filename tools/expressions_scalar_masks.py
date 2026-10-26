import numpy as np

def masked_softmax(x):
    a=np.where(x.astype(np.float64)>0,x.astype(np.float64),-100.)
    e=np.exp(a-a.max(-1,keepdims=True))
    return [(e/e.sum(-1,keepdims=True)).astype(np.float32)]

OPEN=[
 dict(id='p_where_sign',code='x=torch.randn(257); torch.where(x>0,x,-x)',ref=lambda x:[np.abs(x)]),
 dict(id='p_comparison_float',code='(torch.randn(513)>0).float()',ref=lambda x:[(x>0).astype(np.float32)]),
 dict(id='p_where_interval',code='x=torch.randn(129); torch.where((x>-.5)&(x<.5),x,0.)',ref=lambda x:[np.where((x>-.5)&(x<.5),x,np.float32(0))]),
 dict(id='f_positive_sum',code='x=torch.randn(7,129); torch.where(x>0,x,0.).sum(-1)',ref=lambda x:[np.maximum(x.astype(np.float64),0).sum(-1).astype(np.float32)]),
 dict(id='f_masked_softmax',code='x=torch.randn(7,129); torch.softmax(torch.where(x>0,x,-100.),dim=-1)',ref=masked_softmax),
]
for spec in OPEN:spec.update(rtol=1e-4,atol=2e-5)
