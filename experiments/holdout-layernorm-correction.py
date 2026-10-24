"""Correct the callable boundary of the original LayerNorm fixture.

The original expression precomputed y outside tracing, making y the IR input.
This module keeps centering inside forward. No compiler change accompanies it.
"""
import numpy as np


def reference(x):
    y=x.astype(np.float64)-x.astype(np.float64).mean(-1,keepdims=True)
    return [(y/np.sqrt((y*y).mean(-1,keepdims=True)+1e-5)).astype(np.float32)]


OPEN=[dict(id='f_holdout_layernorm_module',code='''
class Normalization(nn.Module):
    def forward(self,x):
        y=x-x.mean(-1,keepdim=True)
        return y/torch.sqrt((y*y).mean(-1,keepdim=True)+1e-5)
Normalization()(torch.randn(5,83))
''',ref=reference,rtol=1e-3,atol=1e-4)]
