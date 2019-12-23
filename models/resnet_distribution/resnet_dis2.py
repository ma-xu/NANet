import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
import time
from torch.distributions.multivariate_normal import MultivariateNormal

__all__ = ['dis2_resnet18', 'dis2_resnet34', 'dis2_resnet50', 'dis2_resnet101', 'dis2_resnet152']

class DisLayer(nn.Module):
    def __init__(self, channel, reduction=16, local_num=8):
        super(DisLayer, self).__init__()
        self.embedding = nn.Conv2d(in_channels=channel,out_channels=local_num,kernel_size=1)
        # self.normal_loc = Parameter(torch.rand(local_num,2)) # 2 means weight, height
        self.normal_scal = Parameter(torch.rand(2))
        self.local_num = local_num
        self.position_scal = Parameter(torch.ones(1))
        self.value_embed = nn.Sequential(
            nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=5,groups=channel,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, groups=channel, padding=2),
        )
        # self.localation_map = self.get_localation_map(1,224,224,1)
        # print(self.localation_map.shape)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):

        b,c,w,h = x.size()
        #Step1: embedding for each local point.
        x_embedded = self.embedding(x)

        # Learning the location. Shape: [b,loc_number,2]
        norm_loc = self.get_max_index(x_embedded)
        # Learning the scale_tril. Shape: [b,loc_number,2,2]
        aa = x_embedded.mean(dim=3).std(dim=-1,keepdim=True)
        bb = x_embedded.mean(dim=2).std(dim=-1,keepdim=True)
        scale_tril = (torch.cat([aa,bb],dim=-1)*self.normal_scal).diag_embed()

        # Step2： Distribution
        # TODO: Learn a local point for each channel.
        multiNorm = MultivariateNormal(loc=norm_loc*self.position_scal,scale_tril=scale_tril)
        localtion_map = self.get_location_mask(x,b,w,h,self.local_num) # shape[w, h, b,local_num, 2]
        pdf = multiNorm.log_prob(localtion_map*self.position_scal).exp() # shape [w,h,b,local_num]
        pdf = pdf.permute(2,0,1,3).unsqueeze(dim=1) #shape [b,1,w,h,local_num]
        pdf[pdf != pdf] = 0 # Remove NaN due to precision.



        #Step3: Value embedding
        x_value = x.expand(self.local_num,b,c,w,h).reshape(self.local_num*b,c,w,h)
        x_value = self.value_embed(x_value).reshape(self.local_num,b,c,w,h).permute(1,2,3,4,0) # shape [b,c,w,h,local_num]


        #Step4: embeded_Value X possibility_density
        increment = F.relu((x_value*pdf).mean(dim=-1),inplace=True)
        increment = torch.rand(x.size()).cuda()
        # print(increment)
        return increment + x

    def get_location_mask(self,x,b,w,h,local_num):
        mask = (x[0, 0, :, :] != -999).nonzero()
        mask = mask.reshape(w, h, 2)
        return mask.expand(b,local_num, w, h, 2).permute(2,3,0,1,4)


    def get_max_index(self,x):
        # x shape: [b,c,w,h]
        index1 = (x.max(2)[0]).max(-1, keepdim=True)[1]
        index2 = (x.max(3)[0]).max(-1, keepdim=True)[1]
        return torch.cat([index2, index1], dim=-1)

    # def get_localation_map(self,b,w,h,local_num):
    #     ww = torch.arange(0, w).view(1, w)
    #     hh = torch.arange(0, h).view(h, 1)
    #     position = torch.broadcast_tensors(ww, hh)
    #     loc_map = torch.cat([position[1].unsqueeze(dim=-1), position[0].unsqueeze(dim=-1)], dim=-1)
    #     return loc_map.expand(b,local_num, w, h, 2).permute(0,2,3,1,4)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dis  = DisLayer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dis(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.dis  = DisLayer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dis(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def dis2_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def dis2_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def dis2_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def dis2_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def dis2_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model




def demo():
    st = time.perf_counter()
    for i in range(100):
        net = dis2_resnet50(num_classes=1000)
        y = net(torch.randn(2, 3, 224,224))
        print("epoch: {},  shape: {}".format(i,y.size()))
    print("CPU time: {}".format(time.perf_counter() - st))

def demo2():
    st = time.perf_counter()
    for i in range(100):
        net = dis2_resnet50(num_classes=1000).cuda()
        y = net(torch.randn(2, 3, 224,224).cuda())
        print("epoch: {},  shape: {}".format(i,y.size()))
        print("Allocated: {}".format(torch.cuda.memory_allocated()/(1e6)))
    print("GPU time: {}".format(time.perf_counter() - st))

# demo()
demo2()
