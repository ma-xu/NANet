import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

__all__ = [  'prm_mnasnet1_0']


# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997

class PRMLayer(nn.Module):
    def __init__(self,channel,groups=8,mode='dotproduct'):
        super(PRMLayer, self).__init__()
        self.mode = mode
        # print(channel)
        self.groups = groups
        self.max_pool = nn.AdaptiveMaxPool2d(1,return_indices=True)
        self.weight = Parameter(torch.zeros(1,self.groups,1,1))
        self.bias = Parameter(torch.ones(1,self.groups,1,1))
        self.sig = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.one = Parameter(torch.ones(1,self.groups,1))
        self.zero = Parameter(torch.zeros(1, self.groups, 1))
        self.theta = Parameter(torch.rand(1,2,1,1))
        self.scale =  Parameter(torch.ones(1))

    def forward(self, x):

        b,c,h,w = x.size()
        position_mask = self.get_position_mask(x, b, h, w, self.groups)
        # Similarity function
        query_value, query_position = self.get_query_position(x, self.groups)  # shape [b*num,2,1,1]
        query_value = query_value.view(b*self.groups,-1,1)
        x_value = x.view(b*self.groups,-1,h*w)
        similarity_max = self.get_similarity(x_value, query_value, mode=self.mode)
        similarity_gap = self.get_similarity(x_value, self.gap(x).view(b*self.groups,-1,1), mode=self.mode)

        similarity_max = similarity_max.view(b,self.groups,h*w)

        Distance = abs(position_mask - query_position)
        Distance = Distance.type(query_value.type())
        # Distance = torch.exp(-Distance * self.theta)
        distribution = Normal(0, self.scale)
        Distance = distribution.log_prob(Distance * self.theta).exp().clone()
        Distance = (Distance.mean(dim=1)).view(b, self.groups, h * w)
        # # add e^(-x), means closer more important
        # Distance = torch.exp(-Distance * self.theta)
        # Distance = (self.distance_embedding(Distance)).reshape(b, self.groups, h*w)
        similarity_max = similarity_max*Distance


        similarity_gap = similarity_gap.view(b, self.groups, h*w)
        similarity = similarity_max*self.zero+similarity_gap*self.one



        context = similarity - similarity.mean(dim=2, keepdim=True)
        std = context.std(dim=2, keepdim=True) + 1e-5
        context = (context/std).view(b,self.groups,h,w)
        # affine function
        context = context * self.weight + self.bias
        context = context.view(b*self.groups,1,h,w)\
            .expand(b*self.groups, c//self.groups, h, w).reshape(b,c,h,w)
        value = x*self.sig(context)

        return value

    def get_position_mask(self,x,b,h,w,number):
        mask = (x[0, 0, :, :] != 2020).nonzero()
        mask = (mask.reshape(h,w, 2)).permute(2,0,1).expand(b*number,2,h,w)
        return mask


    def get_query_position(self, query,groups):
        b,c,h,w = query.size()
        value = query.view(b*groups,c//groups,h,w)
        sumvalue = value.sum(dim=1,keepdim=True)
        maxvalue,maxposition = self.max_pool(sumvalue)
        t_position = torch.cat((maxposition//w,maxposition % w),dim=1)

        t_value = value[torch.arange(b*groups),:,t_position[:,0,0,0],t_position[:,1,0,0]]
        t_value = t_value.view(b, c, 1, 1)
        return t_value, t_position

    def get_similarity(self,query, key_value, mode='dotproduct'):
        if mode == 'dotproduct':
            similarity = torch.matmul(key_value.permute(0, 2, 1), query).squeeze(dim=1)
        elif mode == 'l1norm':
            similarity = -(abs(query - key_value)).sum(dim=1)
        elif mode == 'gaussian':
            # Gaussian Similarity (No recommanded, too sensitive to noise)
            similarity = torch.exp(torch.matmul(key_value.permute(0, 2, 1), query))
            similarity[similarity == float("Inf")] = 0
            similarity[similarity <= 1e-9] = 1e-9
        else:
            similarity = torch.matmul(key_value.permute(0, 2, 1), query)
        return similarity



class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor,
                 bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum))
        self.prm = PRMLayer(out_ch)

    def forward(self, input):
        if self.apply_residual:
            return self.prm(self.layers(input)) + input
        else:
            return self.prm(self.layers(input))


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats,
           bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor,
                              bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor,
                              bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _scale_depths(depths, alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    # """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf.
    # >>> model = MNASNet(1000, 1.0)
    # >>> x = torch.rand(1, 3, 224, 224)
    # >>> y = model(x)
    # >>> y.dim()
    # 1
    # >>> y.nelement()
    # 1000
    # """

    def __init__(self, alpha, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()
        depths = _scale_depths([24, 40, 80, 96, 192, 320], alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(16, depths[0], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[0], depths[1], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[1], depths[2], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2d(depths[5], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)


def prm_mnasnet0_5(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, **kwargs)
    return model


def prm_mnasnet0_75(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, **kwargs)
    return model


def prm_mnasnet1_0(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0, **kwargs)
    return model


def prm_mnasnet1_3(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.3, **kwargs)
    return model

def demo():
    net = prm_mnasnet1_0(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
#
