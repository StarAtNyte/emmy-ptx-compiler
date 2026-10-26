import numpy as np
from scipy.special import erf

def gelu(x):
    a=x.astype(np.float64)
    return [(a*.5*(1+erf(a/np.sqrt(2)))).astype(np.float32)]

OPEN=[
 dict(id='p_square_257',code='torch.square(torch.randn(257))',ref=lambda x:[np.square(x)]),
 dict(id='p_erf_513',code='torch.erf(torch.randn(513))',ref=lambda x:[erf(x.astype(np.float64)).astype(np.float32)]),
 dict(id='f_gelu_7_65',code='torch.nn.functional.gelu(torch.randn(7,65))',ref=gelu),
 dict(id='f_gelu_math_5_129',code='x=torch.randn(5,129); .5*x*(1+torch.erf(x/1.4142135623730951))',ref=gelu),
 dict(id='f_rms_square_9_129',code='x=torch.randn(9,129); x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)',
      ref=lambda x:[(x.astype(np.float64)/np.sqrt((x.astype(np.float64)**2).mean(-1,keepdims=True)+1e-5)).astype(np.float32)]),
 dict(id='p_zero_pad_add_129',code='torch.nn.functional.pad(torch.randn(129),(0,0))+1',ref=lambda x:[x+np.float32(1)]),
]
for spec in OPEN:spec.update(rtol=1e-4,atol=2e-5)
