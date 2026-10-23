"""Larger pointwise workloads for vector-instruction timing."""
import numpy as np

OPEN = [
    dict(id='p_large_add',code='torch.randn(1048576)+torch.randn(1048576)',
         ref=lambda x,y:[x+y],rtol=0,atol=0),
    dict(id='p_large_scale',code='torch.randn(1048576)*2.0+1.0',
         ref=lambda x:[x*np.float32(2)+np.float32(1)],rtol=1e-6,atol=1e-6),
]
