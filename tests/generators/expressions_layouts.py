"""Batched, broadcast and non-last-axis layouts within the five categories."""
import numpy as np


OPEN = []
REDUCE = 'WORK=t128,REDUCE=coop'


def case(name, code, ref, *, rtol=1e-3, atol=1e-4, knobs=''):
    OPEN.append(dict(id=name, code=code, ref=ref, knobs=knobs, rtol=rtol, atol=atol))


def f64(x):
    return x.astype(np.float64)


def out(x):
    return x.astype(np.float32)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


case('ly_p_outer_add', 'x=torch.randn(37); y=torch.randn(53); x[:,None]+y[None,:]', lambda x, y: [x[:, None] + y[None, :]], rtol=0, atol=0)
case('ly_p_row_scale', 'x=torch.randn(24,40); s=torch.randn(24,1); x*s', lambda x, s: [x*s], rtol=0, atol=0)
case('ly_p_3d_bias', 'x=torch.randn(3,5,7); b=torch.randn(7); torch.relu(x+b)', lambda x, b: [np.maximum(x+b, 0)], rtol=0, atol=0)
case('ly_r_axis0', 'torch.randn(300,7).sum(0)', lambda x: [out(f64(x).sum(0))], rtol=1e-4, atol=1e-4, knobs=REDUCE)
case('ly_r_mid_axis', 'torch.randn(4,50,6).sum(1)', lambda x: [out(f64(x).sum(1))], rtol=1e-4, atol=1e-4, knobs=REDUCE)
case('ly_r_3d_last', 'torch.randn(3,4,100).mean(-1)', lambda x: [out(f64(x).mean(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE)
case('ly_r_all_2d', 'torch.randn(30,40).sum()', lambda x: [out(f64(x).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE)
case('ly_f_softmax_3d', 'torch.softmax(torch.randn(2,6,70),dim=-1)', lambda x: [out(softmax(f64(x)))], atol=1e-5)
case('ly_f_softmax_axis0', 'torch.softmax(torch.randn(40,9),dim=0)', lambda x: [out(softmax(f64(x), 0))], atol=1e-5)
case('ly_f_rms_3d', 'x=torch.randn(2,5,64); w=torch.randn(64); x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)*w',
     lambda x, w: [out(f64(x)/np.sqrt((f64(x)**2).mean(-1, keepdims=True)+1e-5)*w)])
case('ly_m_bmm', 'torch.bmm(torch.randn(3,16,24), torch.randn(3,24,8))', lambda a, b: [out(f64(a)@f64(b))], atol=1e-3)
case('ly_m_batched_broadcast', 'torch.randn(4,10,12) @ torch.randn(12,6)', lambda a, b: [out(f64(a)@f64(b))], atol=1e-3)
# Matrix-vector products are omitted: Emmy 0.3.11 traces them to IR that reads
# only the first vector element, so its own reference would not pass either.
case('ly_a_batched', 'q=torch.randn(2,16,8); kt=torch.randn(2,8,16); v=torch.randn(2,16,8); torch.softmax(q@kt/2.8284271247461903,dim=-1)@v',
     lambda q, kt, v: [out(softmax(f64(q)@f64(kt)/np.sqrt(8))@f64(v))])
case('ly_a_heads', 'q=torch.randn(4,32,16); kt=torch.randn(4,16,32); v=torch.randn(4,32,16); torch.softmax(q@kt*0.25,dim=-1)@v',
     lambda q, kt, v: [out(softmax(f64(q)@f64(kt)*0.25)@f64(v))])
case('ly_a_causal', 'q=torch.randn(16,8); kt=torch.randn(8,16); v=torch.randn(16,8); m=torch.triu(torch.ones(16,16),1)*-1e9; torch.softmax(q@kt*0.35+m,dim=-1)@v',
     lambda q, kt, m, v: [out(softmax(f64(q)@f64(kt)*0.35 + f64(m))@f64(v))])
