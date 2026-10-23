"""Schedule diversity, including transposed cooperative reductions."""
import numpy as np

OPEN = []
for width in (32, 64, 256, 512):
    OPEN.append(dict(id=f'r_rows_t{width}',
                     code='torch.randn(11,333).sum(-1)',
                     ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)],
                     knobs=f'WORK=t{width},REDUCE=coop', rtol=1e-3, atol=1e-4))
for width in (64, 256):
    def softmax(x):
        x=x.astype(np.float64)
        e=np.exp(x-x.max(-1,keepdims=True))
        return [(e/e.sum(-1,keepdims=True)).astype(np.float32)]
    OPEN.append(dict(id=f'f_softmax_t{width}',
                     code='torch.softmax(torch.randn(7,513),dim=-1)',
                     ref=softmax, knobs=f'WORK=t{width},REDUCE=coop', rtol=1e-3, atol=1e-5))
for width in (32, 128, 256):
    OPEN.append(dict(id=f'r_columns_transposed_t{width}',
                     code='torch.randn(257,9).sum(0)',
                     ref=lambda x: [x.astype(np.float64).sum(0).astype(np.float32)],
                     knobs=f'WORK=t{width},REDUCE=coop-t', rtol=1e-3, atol=1e-4))
OPEN.append(dict(id='r_rows_serial', code='torch.randn(13,71).sum(-1)',
                 ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)],
                 knobs='REDUCE=', rtol=1e-3, atol=1e-4))
