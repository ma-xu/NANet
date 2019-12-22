from torch.distributions.multivariate_normal import MultivariateNormal
import torch
normal_loc = torch.rand(4,8,2)
normal_scal = torch.rand(4,8,2)
multiNorm = MultivariateNormal(loc=normal_loc,scale_tril=(normal_scal).diag_embed())
