"""Actual GELU MLP, with seeded CPU reference and explicit parameter restoration."""
import torch
from torch import nn


def make_model():
    torch.manual_seed(2026)
    return nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,24),nn.GELU(),nn.Linear(24,8)).eval()


def constants():
    model=make_model();result={}
    for index,name in [(0,'linear_wt'),(2,'linear_1_wt'),(4,'linear_2_wt')]:
        result[name]=model[index].weight.detach().numpy().T.copy()
        result[f'p_layers_{index}_bias']=model[index].bias.detach().numpy().copy()
    return result


def reference(x):
    with torch.no_grad():return [make_model()(torch.from_numpy(x.copy())).numpy()]


OPEN=[dict(id='model_mlp_gelu',code='''
torch.manual_seed(2026)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,24),nn.GELU(),nn.Linear(24,8))
    def forward(self,x):
        return self.layers(x)
MLP().eval()(torch.randn(3,16))
''',ref=reference,constants=constants,rtol=1e-3,atol=1e-4)]
