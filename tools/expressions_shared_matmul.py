"""Independent matmul shapes for target shared-load scheduling experiments."""
import numpy as np


OPEN = []
for m, k, n in ((256, 512, 384), (384, 256, 320), (512, 256, 256), (257, 129, 131)):
    OPEN.append(dict(id=f'm_shared_{m}_{k}_{n}',
                     code=f'torch.randn({m},{k}) @ torch.randn({k},{n})',
                     ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
                     rtol=1e-3, atol=1e-3))
