
import torch
import sys
sys.path.append('../')
import models as models
from collections import OrderedDict
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-p', '--path', default='r1_gc_resnet50', type=str)
args = parser.parse_args()


checkpoint_path ='/Users/melody/Downloads/prm23_resnet50/'
load_path = checkpoint_path + 'model_best.pth.tar'
check_point = torch.load(load_path,map_location=torch.device('cpu'))

# max zero, gap one
one = []
zero = []
for k, v in check_point['state_dict'].items():
    if "prm.one" in k:
        # print(k)
        # print(v.shape)
        v=v.squeeze(dim=0).squeeze(dim=-1)
        one.append(v)
    if "prm.zero" in k:
        # print(k)
        # print(v.shape)
        v=v.squeeze(dim=0).squeeze(dim=-1)
        zero.append(v)
print(one.__len__())
print(zero.__len__())

# 3, 4, 6, 3       2 6 12 15

for block_index in range(16):
    # print(i)
    rate = zero[block_index]/(zero[block_index]+one[block_index])
    # print(rate.numpy())
    print(rate.mean())
# plt.plot(rate)
#
# plt.show()





