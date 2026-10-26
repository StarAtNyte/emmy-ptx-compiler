"""Fresh frontend shapes/compositions for scalar comparison and mask lowering."""
import numpy as np


def masked_attention(q,k,v):
    scores=q.astype(np.float64)@k.astype(np.float64)
    scores=np.where(scores>0,scores,-100.)
    e=np.exp(scores-scores.max(-1,keepdims=True))
    return [((e/e.sum(-1,keepdims=True))@v.astype(np.float64)).astype(np.float32)]


OPEN=[
 dict(id='p_less_equal_513',code='x=torch.randn(513); torch.where(x<=.25,x*2,x-.5)',
      ref=lambda x:[np.where(x<=.25,x*np.float32(2),x-np.float32(.5))]),
 dict(id='p_equal_zero_259',code='x=torch.randn(259); torch.where(torch.relu(x)==0,1.,0.)',
      ref=lambda x:[(x<=0).astype(np.float32)]),
 dict(id='p_not_equal_zero_131',code='x=torch.randn(131); torch.where(torch.relu(x)!=0,1.,0.)',
      ref=lambda x:[(x>0).astype(np.float32)]),
 dict(id='p_mask_or_not_515',code='x=torch.randn(515); (~((x>=-.25)|(x>.75))).float()',
      ref=lambda x:[(~((x>=-.25)|(x>.75))).astype(np.float32)]),
 dict(id='f_negative_sum_9_65',code='x=torch.randn(9,65); torch.where(x<0,x,0.).sum(-1)',
      ref=lambda x:[np.minimum(x.astype(np.float64),0).sum(-1).astype(np.float32)]),
 dict(id='a_masked_attention_16',code='q=torch.randn(16,16); k=torch.randn(16,16); v=torch.randn(16,16); torch.softmax(torch.where(q@k>0,q@k,-100.),dim=-1)@v',ref=masked_attention),
]
for spec in OPEN:spec.update(rtol=1e-3,atol=1e-3)
