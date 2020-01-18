from torch import nn
import torch
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal


__all__ = ['prm_mobilenet_v2']

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



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.prm = PRMLayer(oup)

    def forward(self, x):

        if self.use_res_connect:
            return x + self.prm(self.conv(x))
        else:
            return self.prm(self.conv(x))


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def prm_mobilenet_v2(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    return model


def demo():
    net = prm_mobilenet_v2(num_classes=1000)
    # print(net)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
