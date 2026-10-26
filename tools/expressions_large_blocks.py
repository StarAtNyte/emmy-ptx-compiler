"""Fresh schedule/shape probes for large prescribed cooperative blocks.

References use float64 NumPy and the published category tolerances. These test
frontend-generated IR in addition to the explicit live-register pressure probes.
"""
import numpy as np


OPEN = []


def f32(x):
    return np.asarray(x, dtype=np.float32)


def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return f32(e / e.sum(-1, keepdims=True))


# Emmy 0.3.11 recognizes these work-width presets. Unknown t320/t768/t1024
# strings silently fall back to 256; do not count them as distinct schedules.
for threads in (256, 512):
    for rows, cols in ((3, 67), (5, 1031)):
        knobs = f'WORK=t{threads},REDUCE=coop'
        for op in ('sum', 'amax'):
            OPEN.append(dict(
                id=f'large_{op}_{rows}_{cols}_t{threads}',
                code=f'torch.randn({rows},{cols}).{op}(-1)', knobs=knobs,
                ref=(lambda x: [f32(x.astype(np.float64).sum(-1))]) if op == 'sum'
                    else (lambda x: [x.max(-1)]), rtol=1e-4, atol=1e-4))
        OPEN.append(dict(
            id=f'large_softmax_{rows}_{cols}_t{threads}',
            code=f'torch.softmax(torch.randn({rows},{cols})*7.0,dim=-1)',
            knobs=knobs, ref=lambda x: [softmax(x * np.float32(7))],
            rtol=1e-3, atol=1e-5))
        OPEN.append(dict(
            id=f'large_rms_{rows}_{cols}_t{threads}',
            code=f'x=torch.randn({rows},{cols}); x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)',
            knobs=knobs, ref=lambda x: [f32(x.astype(np.float64) /
                np.sqrt((x.astype(np.float64)**2).mean(-1, keepdims=True)+1e-5))],
            rtol=1e-3, atol=1e-4))
