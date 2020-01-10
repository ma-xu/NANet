from torch.distributions.multivariate_normal import MultivariateNormal
import torch
# normal_loc = torch.rand(4,8,2)
# normal_scal = torch.rand(4,8,2)
# multiNorm = MultivariateNormal(loc=normal_loc,scale_tril=(normal_scal).diag_embed())
import torch.nn as nn
import torchvision.models



net = nn.Conv2d(64,32,1,groups=16)
x=torch.rand(1,64,3,3)
y = net(x)
print(y)


max_pooling2 = nn.AdaptiveMaxPool2d(1,return_indices=True)


x=torch.rand(1,1,3,3)
print(x)
y,index = max_pooling2(x)
print(y)
print(index)
print((x==y).nonzero())


# demo = torch.rand(1,2,3,3)
# demo2 = demo.repeat(1,2,1,1)
# print(demo2)

# norm_loc = (demo == max_pooling(demo)).nonzero()
# norm_loc = (norm_loc.view(2,2,4))[:,:,2:4]
# print(norm_loc)

# mmm,ind = max_pooling2(demo)
# print(ind)


# mmm  =max_pooling

# index1 = (demo.max(2)[0]).max(-1,keepdim=True)[1]
# index2 = (demo.max(3)[0]).max(-1,keepdim=True)[1]
# index = torch.cat([index2,index1],dim=-1)
# print(norm_loc == index)
