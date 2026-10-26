"""Pointwise transcendental corpus for tight-tolerance accuracy measurement.

Each case is one elementwise op over 4096 values. The measurement tool replaces
the generator's random inputs with wide-range arrays and compares against
float64 references, so the tolerances here only gate the default inputs.
"""
import numpy as np


OPEN = []
N = 4096


def case(name, code, ref):
    OPEN.append(dict(id=name, code=code, ref=ref, knobs='', rtol=1e-6, atol=1e-6))


def f64(fn):
    return lambda x: [fn(x.astype(np.float64)).astype(np.float32)]


case('t_tanh', f'torch.tanh(torch.randn({N}))', f64(np.tanh))
case('t_exp', f'torch.exp(torch.randn({N}))', f64(np.exp))
case('t_sigmoid', f'torch.sigmoid(torch.randn({N}))', f64(lambda x: 1/(1+np.exp(-x))))
case('t_silu', f'torch.nn.functional.silu(torch.randn({N}))', f64(lambda x: x/(1+np.exp(-x))))
case('t_log', f'torch.log(torch.rand({N}))', f64(np.log))
case('t_log2', f'torch.log2(torch.rand({N})+0.5)', f64(np.log2))
case('t_rsqrt', f'torch.rsqrt(torch.rand({N}))', f64(lambda x: 1/np.sqrt(x)))
case('t_sqrt', f'torch.sqrt(torch.rand({N}))', f64(np.sqrt))
case('t_reciprocal', f'torch.reciprocal(torch.randn({N}))', f64(lambda x: 1/x))
case('t_gelu_tanh', f"torch.nn.functional.gelu(torch.randn({N}), approximate='tanh')",
     f64(lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))))
case('t_gelu', f'torch.nn.functional.gelu(torch.randn({N}))',
     f64(lambda x: 0.5*x*(1+np.vectorize(__import__('math').erf)(x/np.sqrt(2)))))
case('t_softplus', f'torch.nn.functional.softplus(torch.randn({N}))', f64(lambda x: np.log1p(np.exp(x))))
case('t_expm1', f'torch.expm1(torch.randn({N}))', f64(np.expm1))
case('t_log1p', f'torch.log1p(torch.rand({N}))', f64(np.log1p))
case('t_exp2', f'torch.exp2(torch.randn({N}))', f64(np.exp2))
case('t_elu', f'torch.nn.functional.elu(torch.randn({N}))', f64(lambda x: np.where(x > 0, x, np.expm1(x))))
case('t_mish', f'torch.nn.functional.mish(torch.randn({N}))', f64(lambda x: x*np.tanh(np.log1p(np.exp(x)))))
case('t_pow3', f'torch.randn({N})**3', f64(lambda x: x**3))
case('t_pow_half', f'torch.rand({N})**0.5', f64(np.sqrt))
case('t_sin', f'torch.sin(torch.randn({N}))', f64(np.sin))
case('t_cos', f'torch.cos(torch.randn({N}))', f64(np.cos))
case('t_hardswish', f'torch.nn.functional.hardswish(torch.randn({N}))', f64(lambda x: x*np.clip(x+3, 0, 6)/6))
case('t_leaky', f'torch.nn.functional.leaky_relu(torch.randn({N}), 0.1)', f64(lambda x: np.where(x > 0, x, 0.1*x)))
case('t_clamp', f'torch.clamp(torch.randn({N}), -0.5, 0.5)', f64(lambda x: np.clip(x, -0.5, 0.5)))
case('t_abs_neg', f'-torch.abs(torch.randn({N}))', f64(lambda x: -np.abs(x)))
