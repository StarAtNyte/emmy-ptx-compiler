"""A complete, small pre-norm attention/MLP residual block for validation."""
import ast
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(16)
        self.q = nn.Linear(16, 16, bias=False)
        self.k = nn.Linear(16, 16, bias=False)
        self.v = nn.Linear(16, 16, bias=False)
        self.out = nn.Linear(16, 16, bias=False)
        self.up = nn.Linear(16, 24)
        self.down = nn.Linear(24, 16)

    def forward(self, x):
        z = self.norm(x)
        attention = F.scaled_dot_product_attention(self.q(z), self.k(z), self.v(z))
        y = x + self.out(attention)
        return y + self.down(F.silu(self.up(y)))


def make_block():
    torch.manual_seed(2027)
    return Block().eval()


_source = Path(__file__).read_text()
_class = next(n for n in ast.parse(_source).body if isinstance(n, ast.ClassDef) and n.name == 'Block')
PREFIX = 'torch.manual_seed(2027)\n' + ast.get_source_segment(_source, _class) + '\nmodel = Block().eval()\n'


def stage_reference(name):
    def reference(*arrays):
        model = make_block()
        with torch.no_grad():
            tensors = [torch.from_numpy(a.copy()) for a in arrays]
            if name == 'attention':
                result = F.scaled_dot_product_attention(*tensors)
            elif name == 'residual_ff':
                y = tensors[0] + tensors[1]
                result = y + model.down(F.silu(model.up(y)))
            else:
                result = getattr(model, name)(tensors[0])
        return [result.numpy()]
    return reference


def stage_constants(name):
    def constants():
        model = make_block()
        if name == 'norm':
            return {'p_weight': model.norm.weight.detach().numpy()}
        if name == 'residual_ff':
            return {'linear_wt': model.up.weight.detach().numpy().T,
                    'linear_1_wt': model.down.weight.detach().numpy().T,
                    'p_up_bias': model.up.bias.detach().numpy(),
                    'p_down_bias': model.down.bias.detach().numpy()}
        if name == 'attention':
            return {}
        return {'linear_wt': getattr(model, name).weight.detach().numpy().T}
    return constants


OPEN = []
for name in ('norm', 'q', 'k', 'v', 'out'):
    OPEN.append(dict(id='transformer_' + name,
                     code=PREFIX + f'model.{name}(torch.randn(1,4,16))',
                     ref=stage_reference(name), constants=stage_constants(name),
                     rtol=1e-3, atol=1e-4))
OPEN.append(dict(id='transformer_attention',
                 code='q=torch.randn(1,4,16); k=torch.randn(1,4,16); v=torch.randn(1,4,16); F.scaled_dot_product_attention(q,k,v)',
                 ref=stage_reference('attention'), constants=stage_constants('attention'),
                 rtol=1e-3, atol=1e-4))
OPEN.append(dict(id='transformer_residual_ff', code=PREFIX + '''
class ResidualFF(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.up=model.up
        self.down=model.down
    def forward(self,x,a):
        y=x+a
        return y+self.down(F.silu(self.up(y)))
ResidualFF(model).eval()(torch.randn(1,4,16),torch.randn(1,4,16))
''', ref=stage_reference('residual_ff'), constants=stage_constants('residual_ff'),
                 rtol=1e-3, atol=1e-4))
