"""Small actual nn.Module models, checked against CPU PyTorch.

Parameter initialization is seeded in both the traced model and the reference.
The input uses the generator's independent NumPy seed.
"""
import torch
from torch import nn


def make_mlp():
    torch.manual_seed(2026)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 24),
                         nn.SiLU(), nn.Linear(24, 8)).eval()


def mlp_constants():
    model = make_mlp()
    result = {}
    for index, weight_name in ((0, 'linear_wt'), (2, 'linear_1_wt'), (4, 'linear_2_wt')):
        result[weight_name] = model[index].weight.detach().numpy().T.copy()
        result[f'p_layers_{index}_bias'] = model[index].bias.detach().numpy().copy()
    return result


def mlp_reference(*arrays):
    model = make_mlp()
    with torch.no_grad():
        return [model(torch.from_numpy(arrays[0].copy())).numpy()]


OPEN = [dict(
    id='model_mlp_relu_silu',
    code='''
torch.manual_seed(2026)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Linear(32,24), nn.SiLU(), nn.Linear(24,8))
    def forward(self, x):
        return self.layers(x)
MLP().eval()(torch.randn(3,16))
''',
    ref=mlp_reference, constants=mlp_constants, rtol=1e-3, atol=1e-3,
)]
