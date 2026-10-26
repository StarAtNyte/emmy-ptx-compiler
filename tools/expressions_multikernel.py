"""Unpinned expressions that Emmy may schedule as several chained kernels.

The official generator and submit.py chain kernels by buffer name. Shapes and
op mixes stay within the five published categories; no schedule knobs are set,
so Emmy chooses splits, atomics and fusion boundaries itself.
"""
import numpy as np


OPEN = []


def case(name, code, ref, *, rtol=1e-3, atol=1e-3, knobs=''):
    OPEN.append(dict(id=name, code=code, ref=ref, knobs=knobs, rtol=rtol, atol=atol))


def f64(x):
    return x.astype(np.float64)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


for size in (65536, 262144, 1048576):
    case(f'mk_sum_{size}', f'torch.randn({size}).sum()', lambda x: [f64(x).sum().astype(np.float32)])
case('mk_mean_big', 'torch.randn(524288).mean()', lambda x: [f64(x).mean().astype(np.float32)])
case('mk_max_big', 'torch.randn(300000).amax()', lambda x: [x.max()])
case('mk_row_sum_wide', 'torch.randn(4, 65536).sum(-1)', lambda x: [f64(x).sum(-1).astype(np.float32)])
case('mk_col_sum', 'torch.randn(512, 64).sum(0)', lambda x: [f64(x).sum(0).astype(np.float32)])
case('mk_softmax_wide', 'torch.softmax(torch.randn(4, 16384), dim=-1)',
     lambda x: [softmax(f64(x)).astype(np.float32)], atol=1e-5)
case('mk_softmax_tall', 'torch.softmax(torch.randn(2048, 64), dim=-1)',
     lambda x: [softmax(f64(x)).astype(np.float32)], atol=1e-5)
case('mk_rms_wide', 'x=torch.randn(4,16384); w=torch.randn(16384); x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)*w',
     lambda x, w: [(f64(x)/np.sqrt((f64(x)**2).mean(-1, keepdims=True)+1e-5)*w).astype(np.float32)], atol=1e-4)
case('mk_layernorm', 'x=torch.randn(16,1024); m=x.mean(-1,keepdim=True); v=((x-m)*(x-m)).mean(-1,keepdim=True); (x-m)*torch.rsqrt(v+1e-5)',
     lambda x: [((f64(x)-f64(x).mean(-1, keepdims=True))/np.sqrt(f64(x).var(-1, keepdims=True)+1e-5)).astype(np.float32)], atol=1e-4)
case('mk_matmul_sum', 'torch.mm(torch.randn(64,128), torch.randn(128,96)).sum()',
     lambda a, b: [(f64(a)@f64(b)).sum().astype(np.float32)], rtol=1e-3, atol=1e-2)
case('mk_matmul_relu', 'torch.relu(torch.mm(torch.randn(128,256), torch.randn(256,64)))',
     lambda a, b: [np.maximum(f64(a)@f64(b), 0).astype(np.float32)])
case('mk_matmul_bias_gelu', 'x=torch.randn(64,128); w=torch.randn(128,256); b=torch.randn(256); torch.nn.functional.gelu(x@w+b)',
     lambda x, w, b: [(lambda h: 0.5*h*(1+np.vectorize(__import__('math').erf)(h/np.sqrt(2))))(f64(x)@f64(w)+b).astype(np.float32)], atol=1e-3)
case('mk_matmul_tall_k', 'torch.mm(torch.randn(32,4096), torch.randn(4096,32))',
     lambda a, b: [(f64(a)@f64(b)).astype(np.float32)], atol=1e-2)
case('mk_matmul_softmax', 'torch.softmax(torch.mm(torch.randn(64,64), torch.randn(64,512)), dim=-1)',
     lambda a, b: [softmax(f64(a)@f64(b)).astype(np.float32)], atol=1e-5)
case('mk_two_matmul', 'x=torch.randn(64,128); w1=torch.randn(128,128); w2=torch.randn(128,64); torch.relu(x@w1)@w2',
     lambda x, w1, w2: [(np.maximum(f64(x)@f64(w1), 0)@f64(w2)).astype(np.float32)], atol=1e-2)
case('mk_attention_long', 'q=torch.randn(64,32); kt=torch.randn(32,1024); v=torch.randn(1024,32); torch.softmax(q@kt/5.656854249492381,dim=-1)@v',
     lambda q, kt, v: [(softmax(f64(q)@f64(kt)/np.sqrt(32))@f64(v)).astype(np.float32)])
case('mk_attention_wide', 'q=torch.randn(128,64); kt=torch.randn(64,128); v=torch.randn(128,64); torch.softmax(q@kt/8.0,dim=-1)@v',
     lambda q, kt, v: [(softmax(f64(q)@f64(kt)/8)@f64(v)).astype(np.float32)])
case('mk_pointwise_then_sum', 'x=torch.randn(65536); (torch.tanh(x)*torch.sigmoid(x)).sum()',
     lambda x: [(np.tanh(f64(x))/(1+np.exp(-f64(x)))).sum().astype(np.float32)], atol=1e-2)
case('mk_l2_wide', 'x=torch.randn(8,8192); x*torch.rsqrt((x*x).sum(-1,keepdim=True))',
     lambda x: [(f64(x)/np.sqrt((f64(x)**2).sum(-1, keepdims=True))).astype(np.float32)], atol=1e-5)
