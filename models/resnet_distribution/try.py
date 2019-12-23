from torch.distributions.multivariate_normal import MultivariateNormal
import torch
# normal_loc = torch.rand(4,8,2)
# normal_scal = torch.rand(4,8,2)
# multiNorm = MultivariateNormal(loc=normal_loc,scale_tril=(normal_scal).diag_embed())
import torch.nn as nn

max_pooling = nn.AdaptiveMaxPool2d(1)
max_pooling2 = nn.AdaptiveMaxPool2d(1,return_indices=True)




demo = torch.rand(1,2,3,3)
demo2 = demo.repeat(1,2,1,1)
print(demo2)

# norm_loc = (demo == max_pooling(demo)).nonzero()
# norm_loc = (norm_loc.view(2,2,4))[:,:,2:4]
# print(norm_loc)

# mmm,ind = max_pooling2(demo)
# print(ind)


# mmm  =max_pooling

index1 = (demo.max(2)[0]).max(-1,keepdim=True)[1]
index2 = (demo.max(3)[0]).max(-1,keepdim=True)[1]
index = torch.cat([index2,index1],dim=-1)
print(norm_loc == index)
