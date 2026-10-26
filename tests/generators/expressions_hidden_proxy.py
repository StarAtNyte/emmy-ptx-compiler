"""Fresh shape and op-mix variations across the five published categories.

A proxy for the hidden set: same generator, new shapes (odd, prime, tall,
wide), and op mixes Emmy's renderer supports. References are independent
float64 NumPy; tolerances follow the open set's per-category values.
"""
import math

import numpy as np


OPEN = []
REDUCE = 'WORK=t128,REDUCE=coop'
ERF = np.vectorize(math.erf)


def case(name, code, ref, *, rtol=1e-3, atol=1e-4, knobs=''):
    OPEN.append(dict(id=name, code=code, ref=ref, knobs=knobs, rtol=rtol, atol=atol))


def f64(x):
    return x.astype(np.float64)


def out(x):
    return x.astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# Every computation sits inside the traced expression: a separate statement
# such as `x=torch.randn(n)*3.0` would become a runner-seeded input instead.

# 1. Pointwise: exact and tight tolerances, mixed ops, odd sizes.
case('hp_p_add3_1023', 'torch.randn(1023)+torch.randn(1023)+torch.randn(1023)',
     lambda a, b, c: [(a + b) + c], rtol=0, atol=0)
case('hp_p_mul_sub_4099', 'x=torch.randn(4099); y=torch.randn(4099); x*y-x',
     lambda x, y: [x*y - x], rtol=1e-6, atol=1e-6)
case('hp_p_where_2000', 'x=torch.randn(2000); torch.where(x>0, x*2.0, x*0.5)',
     lambda x: [np.where(x > 0, x*np.float32(2), x*np.float32(.5))], rtol=0, atol=0)
case('hp_p_clamp_abs_777', 'torch.clamp(torch.abs(torch.randn(777))-0.5, 0.0, 1.0)',
     lambda x: [np.clip(np.abs(x) - np.float32(.5), 0, 1)], rtol=0, atol=0)
case('hp_p_silu_3000', 'torch.nn.functional.silu(torch.randn(3000))',
     lambda x: [out(f64(x)/(1+np.exp(-f64(x))))], rtol=1e-5, atol=1e-6)
case('hp_p_gelu_tanh_2048', "torch.nn.functional.gelu(torch.randn(2048), approximate='tanh')",
     lambda x: [out(0.5*f64(x)*(1+np.tanh(np.sqrt(2/np.pi)*(f64(x)+0.044715*f64(x)**3))))], rtol=1e-5, atol=1e-6)
case('hp_p_exp_scaled_1500', 'torch.exp(torch.randn(1500)*4.0)',
     lambda x: [out(np.exp(f64(x*np.float32(4))))], rtol=1e-5, atol=1e-6)
case('hp_p_tanh_mix_999', 'x=torch.randn(999); torch.tanh(x)*x+torch.sigmoid(x)',
     lambda x: [out(np.tanh(f64(x))*f64(x) + 1/(1+np.exp(-f64(x))))], rtol=1e-5, atol=1e-6)
case('hp_p_sin_cos_1201', 'x=torch.randn(1201); torch.sin(x*3.0)*torch.cos(x*3.0)',
     lambda x: [out(np.sin(f64(x*np.float32(3)))*np.cos(f64(x*np.float32(3))))], rtol=1e-5, atol=1e-6)
case('hp_p_pow_sqrt_640', 'x=torch.randn(640); torch.sqrt(x*x+1.0)**3',
     lambda x: [out(np.sqrt(f64(x*x+np.float32(1)))**3)], rtol=1e-5, atol=1e-6)
case('hp_p_log_softplus_900', 'x=torch.randn(900); torch.log(torch.exp(x)+1.0)',
     lambda x: [out(np.log(np.exp(f64(x))+1))], rtol=1e-5, atol=1e-6)
case('hp_p_rsqrt_div_1111', 'x=torch.randn(1111); y=torch.randn(1111); x*torch.rsqrt(y*y+1.0)/(x*x+2.0)',
     lambda x, y: [out(f64(x)/np.sqrt(f64(y)**2+1)/(f64(x)**2+2))], rtol=1e-5, atol=1e-6)
case('hp_p_2d_bias_relu', 'x=torch.randn(37,129); b=torch.randn(129); torch.relu(x+b)',
     lambda x, b: [np.maximum(x + b, 0)], rtol=0, atol=0)
case('hp_p_erf_gelu_4096', 'torch.nn.functional.gelu(torch.randn(4096))',
     lambda x: [out(0.5*f64(x)*(1+ERF(f64(x)/np.sqrt(2))))], rtol=1e-5, atol=1e-6)
# 2. Reductions with pinned cooperative schedules.
for size in (1, 33, 1000, 12345, 70001):
    case(f'hp_r_sum_{size}', f'torch.randn({size}).sum()',
         lambda x: [out(f64(x).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE)
case('hp_r_amax_rows', 'torch.randn(13, 999).amax(-1)', lambda x: [x.max(-1)], rtol=0, atol=0, knobs=REDUCE)
case('hp_r_amin_rows', 'torch.randn(6, 300).amin(-1)', lambda x: [x.min(-1)], rtol=0, atol=0, knobs=REDUCE)
case('hp_r_mean_rows', 'torch.randn(31, 517).mean(-1)', lambda x: [out(f64(x).mean(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE)
case('hp_r_sumsq', 'x=torch.randn(5000); (x*x).sum()', lambda x: [out((f64(x)**2).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE)
case('hp_r_abs_rows', 'torch.abs(torch.randn(9, 2047)).sum(-1)', lambda x: [out(np.abs(f64(x)).sum(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE)
case('hp_r_unpinned_rows', 'torch.randn(64, 4096).sum(-1)', lambda x: [out(f64(x).sum(-1))], rtol=1e-4, atol=1e-4)
# 3. Matmul shapes.
for m, k, n in ((1, 64, 1), (7, 13, 11), (100, 50, 70), (256, 256, 256), (33, 1024, 65), (512, 32, 512)):
    case(f'hp_m_{m}_{k}_{n}', f'torch.randn({m},{k}) @ torch.randn({k},{n})',
         lambda a, b: [out(f64(a)@f64(b))], atol=1e-3)
case('hp_m_bias', 'x=torch.randn(96,80); w=torch.randn(80,48); b=torch.randn(48); x@w+b',
     lambda x, w, b: [out(f64(x)@f64(w)+b)], atol=1e-3)
# 4. Fused reductions + elementwise.
case('hp_f_softmax_odd', 'torch.softmax(torch.randn(19, 333), dim=-1)', lambda x: [out(softmax(f64(x)))], atol=1e-5)
case('hp_f_softmax_scaled', 'torch.softmax(torch.randn(8, 512)*3.0, dim=-1)', lambda x: [out(softmax(f64(x*np.float32(3))))], atol=1e-5)
case('hp_f_logsoftmax', 'torch.log_softmax(torch.randn(12, 200), dim=-1)',
     lambda x: [out(f64(x) - f64(x).max(-1, keepdims=True) - np.log(np.exp(f64(x) - f64(x).max(-1, keepdims=True)).sum(-1, keepdims=True)))])
case('hp_f_rms_odd', 'x=torch.randn(11,300); w=torch.randn(300); x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-6)*w',
     lambda x, w: [out(f64(x)/np.sqrt((f64(x)**2).mean(-1, keepdims=True)+1e-6)*w)])
case('hp_f_layernorm_inline', 'x=torch.randn(16,384); (x-x.mean(-1,keepdim=True))*torch.rsqrt(((x-x.mean(-1,keepdim=True))*(x-x.mean(-1,keepdim=True))).mean(-1,keepdim=True)+1e-5)',
     lambda x: [out((f64(x)-f64(x).mean(-1, keepdims=True))/np.sqrt(f64(x).var(-1, keepdims=True)+1e-5))])
case('hp_f_l2_rows', 'x=torch.randn(21,130); x/torch.sqrt((x*x).sum(-1,keepdim=True))',
     lambda x: [out(f64(x)/np.sqrt((f64(x)**2).sum(-1, keepdims=True)))])
case('hp_f_center', 'x=torch.randn(17,250); x-x.mean(-1,keepdim=True)', lambda x: [out(f64(x)-f64(x).mean(-1, keepdims=True))])
# 5. Attention-like.
for seq, depth in ((8, 8), (24, 40), (64, 64), (100, 16), (256, 32)):
    def attention(q, kt, v, depth=depth):
        return [out(softmax(f64(q)@f64(kt)/np.sqrt(depth))@f64(v))]
    case(f'hp_a_{seq}_{depth}',
         f'q=torch.randn({seq},{depth}); kt=torch.randn({depth},{seq}); v=torch.randn({seq},{depth}); torch.softmax(q@kt/{float(np.sqrt(depth))},dim=-1)@v',
         attention)
case('hp_a_rect', 'q=torch.randn(48,32); kt=torch.randn(32,80); v=torch.randn(80,24); torch.softmax(q@kt*0.125,dim=-1)@v',
     lambda q, kt, v: [out(softmax(f64(q)@f64(kt)*0.125)@f64(v))])
