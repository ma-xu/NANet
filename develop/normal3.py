import time
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

w,h = 3,4
local1 = torch.tensor([[0,0],[1,1]])
local2 = torch.tensor([[0,1],[0,1]])
print(local1.shape)
print(local1)


# w,h=3,4
# ww = torch.arange(0,w).view(1, w)
# hh = torch.arange(0,h).view(h,1)
# position = torch.broadcast_tensors(ww,hh)
# loc_map = torch.cat([position[1].unsqueeze(dim=-1),position[0].unsqueeze(dim=-1)],dim=-1)
# print(loc_map.shape)
# print(loc_map)


# b = 4
# scal = torch.rand(b,2).diag_embed()
# loc = torch.zeros(b,2)
# multinorm = MultivariateNormal(loc, scale_tril=scal)
# print(multinorm)
# y = multinorm.log_prob(torch.rand(2)).exp()
# print(y)


# # print(m2.covariance_matrix)
# print(m2.covariance_matrix.shape)
# print(m2.scale_tril.shape)
# y = m2.log_prob(torch.zeros(2))
# print("Batch Shape: {}".format(m2.batch_shape))
# print(y)
