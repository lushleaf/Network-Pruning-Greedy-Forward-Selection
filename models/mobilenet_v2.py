import torch.nn as nn
import math
import torch
import numpy as np


__all__ = ['mobilenetv2']

eps = 1e-40 # subgradient of ReLU in pytorch = 0


# mask layer
class Mask(nn.Module):
    def __init__(self, D_in, layer_num=-1):
        super(Mask, self).__init__()

        '''
        [(a_i + gamma_i)(1 + u_i * gamma) + w_i * gamma] * neuron
        '''

        self.prune_a = nn.Parameter(1./D_in * torch.ones(1, D_in, 1, 1), requires_grad=False)
        self.prune_gamma = nn.Parameter(0. * torch.ones(1, D_in, 1, 1), requires_grad=False)
        #self.prune_u = nn.Parameter(0. * torch.ones(1, D_in, 1, 1), requires_grad=False)
        self.prune_w = nn.Parameter(0. * torch.ones(1, D_in, 1, 1), requires_grad=False)
        self.prune_lsearch = nn.Parameter(0. * torch.tensor(1.), requires_grad=False)
        self.scale = D_in

        self.layer_num = layer_num
        self.D_in = D_in
        self.device = 'cuda'
        self.mode = 'train'
        self.zeros = nn.Parameter(torch.zeros(1,1,1,1), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(1, self.D_in, 1, 1), requires_grad=False)

    def forward(self, x):
        return torch.mul(self.scale * x, ((self.prune_a + self.prune_gamma) * (1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w))

    def pforward(self, x, chosen_layer):
        if self.layer_num == chosen_layer:
            return torch.mul(self.scale * x, ((self.prune_a + self.prune_gamma) * (
                    1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w)), x
        else:
            return torch.mul(self.scale * x, ((self.prune_a + self.prune_gamma) * (
                1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w)), self.zeros

    def turn_off(self, src_param, is_lsearch = False):
        if not is_lsearch:
            tar_param = nn.Parameter(torch.zeros(1, self.D_in, 1, 1), requires_grad=False)
        else:
            tar_param = nn.Parameter(torch.tensor(1.), requires_grad=False)
        tar_param.data = src_param.data.clone()

        return tar_param
    def switch_mode(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.prune_gamma = self.turn_off(self.prune_gamma)
            self.prune_lsearch = self.turn_off(self.prune_lsearch, True)
            self.prune_a = self.turn_off(self.prune_a)

        elif mode == 'prune':
            self.prune_gamma.requires_grad = True
            self.prune_lsearch.requires_grad = True
            self.prune_a = self.turn_off(self.prune_a)

        elif mode == 'adjust_a':
            self.prune_gamma = self.turn_off(self.prune_gamma)
            self.prune_lsearch = self.turn_off(self.prune_lsearch, True)
            self.prune_a.requires_grad = True
        else:
            raise NotImplementedError

    def empty_all_eps(self):
        self.prune_a.data = -eps * self.prune_a.data

    def init_lsearch(self, neuron_index):
        self.prune_gamma.data = 0. * self.prune_gamma.data
        self.prune_w.data = 0.* self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        if neuron_index >= 0:
            self.prune_w[:, neuron_index, :, :] += 1.

    def update_alpha(self, neuron_index, lsearch):
        self.prune_a.data *= (1. - lsearch)
        self.prune_a[:, neuron_index, : ,:] += lsearch
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data
    
    def update_alpha_back(self, neuron_index, lsearch):
        #self.prune_a.data *= (1. - lsearch)
        self.prune_a[:, neuron_index, : ,:] *= 0
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data

    #def set_alpha_to_init(self):
    #    self.prune_a.data = 0. * self.prune_a.data
    #    self.prune_a.data += 1./self.D_in * self.ones#* torch.ones(1, self.D_in, 1, 1).to(self.device)
    #    self.prune_w.data = 0. * self.prune_w.data
    #    self.prune_lsearch.data = 0. * self.prune_lsearch.data
    #    self.prune_gamma.data = 0. * self.prune_gamma.data

    def set_alpha_to_init(self, prunable_neuron):
        if len(prunable_neuron) != self.prune_a.shape[1]:
            print('dim of prunable_neuron error!')
            raise ValueError

        self.prune_a.data = 0. * self.prune_a.data

        num_prunable_neuron = prunable_neuron.sum()
        for _ in range(len(prunable_neuron)):
            if prunable_neuron[_] > 0:
                self.prune_a.data[0, _, 0, 0] += 1. / num_prunable_neuron

        # self.prune_a.data += 1./self.D_in * torch.ones(1, self.D_in, 1, 1).to(self.device)
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data

    def assign_alpha(self, alpha):
        self.prune_a.data = 0. * self.prune_a.data
        self.prune_a.data += alpha
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data

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


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                #
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., dropout=0.2):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        '''
        self.cfgs = [
                            # t, c, n, s
        [1,  16, 1, 1],
        [6,  24, 2, 1],
        [6,  32, 3, 2],
        [6,  16, 4, 2],
        [6,  24, 3, 1],
        [6, 20, 3, 2],
        [6, 40, 1, 1],
        ]
        '''

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)


class Block_prescreen(nn.Module):
    def __init__(self, block, layer_num, skip = False):
        super(Block_prescreen, self).__init__()

        self.conv1 = block[0]
        self.bn1 = block[1]
        self.act1 = block[2]
        self.skip = skip

        if not skip:
            self.mask1 = Mask(D_in=self.conv1.weight.size()[0], layer_num = layer_num)
            self.mask_list = [self.mask1]
        else:
            self.mask_list = []

    def pforward(self, x, score, chosen_layer):
        out = self.act1(self.bn1(self.conv1(x)))
        if not self.skip:
            out, score1 = self.mask1.pforward(out, chosen_layer)
            return out, score + score1, chosen_layer
        else:
            return out, score , chosen_layer


    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        if not self.skip:
            out = self.mask1(out)
        return out


class InvertedResidual_prescreen(nn.Module):
    def __init__(self, block, layer_num):
        super(InvertedResidual_prescreen, self).__init__()
        self.lm = layer_num
        if block.identity:
            self.identity = 1
        else:
            self.identity = 0

        if 1:
            self.conv1 = block.conv[0]
            self.bn1 = block.conv[1]
            self.act1 = block.conv[2]

            #self.mask1 = Mask(D_in=self.conv1.weight.size()[0],layer_num = layer_num)

            self.conv2 = block.conv[3]
            self.bn2 = block.conv[4]
            self.act2 = block.conv[5]
            self.mask1 = Mask(D_in=self.conv2.weight.size()[0],layer_num = layer_num)
            self.conv3 = block.conv[6]
            self.bn3 = block.conv[7]

            self.mask2 = Mask(D_in=self.conv3.weight.size()[0], layer_num=layer_num + 1)

            self.mask_list = [self.mask1, self.mask2]

    def pforward(self, x, score, chosen_layer):
        out = self.act1(self.bn1(self.conv1(x)))
        #out = self.mask1.pforward(out, chosen_layer)
        out = self.act2(self.bn2(self.conv2(out)))
        out, score1 = self.mask1.pforward(out, chosen_layer)
        out = self.bn3(self.conv3(out))
        out, score2 = self.mask2.pforward(out, chosen_layer)

        if self.identity:
            return x + out, score + score1 + score2, chosen_layer
        else:
            return out, score + score1 + score2 , chosen_layer


    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        #out = self.mask1(out)
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.mask1(out)
        out = self.bn3(self.conv3(out))
        out = self.mask2(out)

        if self.identity:
            return x + out
        else:
            return out


class combine_layer(nn.Module):
    def __init__(self, block1, block2, identity):
        super(combine_layer, self).__init__()
        layers = []
        for layer in block1:
            layers.append(layer)
        for layer in block2:
            layers.append(layer)

        self.identity = identity
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_prescreen(nn.Module):
    def __init__(self, net):
        super(MobileNetV2_prescreen, self).__init__()

        # first two layers
        layer_num = 0
        block = combine_layer(net.features[0], net.features[1].conv, net.features[1].identity)
        blocks = [InvertedResidual_prescreen(block, layer_num)]
        layer_num += 2

        for idx, block in enumerate(net.features):
            if idx >=2:
                if isinstance(block, InvertedResidual):
                    block_mask = InvertedResidual_prescreen(block, layer_num)
                    layer_num += 2
                    blocks.append(block_mask)
                else:
                    block_mask = Block_prescreen(block, layer_num)
                    layer_num += 1
                    blocks.append(block_mask)

        blocks.append(Block_prescreen(net.conv, layer_num))
        blocks = nn.Sequential(*blocks)
        layer_num += 1
        self.features = blocks
        self.avgpool = net.avgpool
        self.dropout = net.dropout
        self.classifier = net.classifier


    def pforward(self, x, chosen_layer = -1):
        score = 0.
        for block in self.features:
            x, score, chosen_layer = block.pforward(x, score, chosen_layer)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x, score

    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------- #
# Flops calculation


class tem_block(nn.Module):
    def __init__(self, layers, identity, ind, convd, outd):
        super(tem_block, self).__init__()
        self.conv = nn.Sequential(*layers)
        self.identity = identity
        self.ind = ind
        self.convd = convd
        self.outd = outd  # ind, convd <= outd

    def forward(self, x):
        xx = self.conv(x)
        if self.identity:
            if self.ind < self.outd:
                x = torch.cat([x, torch.zeros(x.size(0), self.outd - self.ind, x.size(2), x.size(3))], 1)
            if self.convd < self.outd:
                xx = torch.cat([xx, torch.zeros(xx.size(0), self.outd - self.convd, xx.size(2), xx.size(3))], 1)
            return x + xx
        else:
            return xx


def build_tem_block(block, ind_idx):
    if isinstance(block, Block_prescreen): # last feature layer
        ind = ind_idx.sum()
        num_c1 = int((block.mask1.prune_a.cpu().squeeze().numpy() > 0.).sum())
        layers = []
        layer = block.conv1
        kernel_size, stride, padding = layer.kernel_size, layer.stride, layer.padding
        layers.append(nn.Conv2d(ind, num_c1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(num_c1))
        layers.append(nn.ReLU6(inplace=True))

        fin_block = tem_block(layers, 0, 0, 0, 0) # no short cut for last feature layer

        return fin_block, (block.mask1.prune_a.cpu().squeeze().numpy() > 0.)

    elif isinstance(block, InvertedResidual_prescreen):

        identity = block.identity
        num_c1 = int((block.mask1.prune_a.cpu().squeeze().numpy() > 0.).sum())
        num_c2 = int((block.mask2.prune_a.cpu().squeeze().numpy() > 0.).sum())

        ind = ind_idx.sum()

        layers = []

        # Conv2d
        layer = block.conv1
        kernel_size, stride, padding = layer.kernel_size, layer.stride, layer.padding
        layers.append(nn.Conv2d(ind, num_c1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))

        # BatchNorm2d and ReLU6
        layers.append(nn.BatchNorm2d(num_c1))
        layers.append(nn.ReLU6(inplace=True))

        # Conv2d
        layer = block.conv2
        kernel_size, stride, padding = layer.kernel_size, layer.stride, layer.padding
        layers.append(nn.Conv2d(num_c1, num_c1, kernel_size=kernel_size, stride=stride, padding=padding, groups=num_c1, bias=False))

        # BatchNorm2d and ReLU6
        layers.append(nn.BatchNorm2d(num_c1))
        layers.append(nn.ReLU6(inplace=True))

        # Conv2d
        layer = block.conv3
        kernel_size, stride, padding = layer.kernel_size, layer.stride, layer.padding
        layers.append(nn.Conv2d(num_c1, num_c2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))

        # BatchNorm2d
        layers.append(nn.BatchNorm2d(num_c2))

        if identity:
            out_idx = ((ind_idx + (block.mask2.prune_a.cpu().squeeze().numpy() > 0.)) > 0)
        else:
            out_idx = (block.mask2.prune_a.cpu().squeeze().numpy() > 0.)


        fin_block = tem_block(layers, identity, ind, num_c2, out_idx.sum())

        return fin_block, out_idx

    else:
        return None

class tem_MobileNetV2(nn.Module):
    def __init__(self, net):
        super(tem_MobileNetV2, self).__init__()
        blocks = []
        ind = (np.ones(3)>0)
        out_features = 1000
        if isinstance(net, nn.DataParallel):
            for block in net.module.features:
                tem_block, ind = build_tem_block(block, ind)
                blocks.append(tem_block)

            self.features = nn.Sequential(*blocks)
            self.avgpool = net.module.avgpool
            self.classifier = nn.Linear(in_features=ind.sum(), out_features=out_features, bias=True)
        else:
            for block in net.features:
                tem_block, ind = build_tem_block(block, ind)
                blocks.append(tem_block)

            self.features = nn.Sequential(*blocks)
            self.avgpool = net.avgpool
            self.classifier = nn.Linear(in_features=ind.sum(), out_features=out_features, bias=True)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mb2_prune_ratio(net_mask):
    net = MobileNetV2()
    from ptflops import get_model_complexity_info
    net.eval()
    fullflops, fullparams = get_model_complexity_info(net, (3, 224, 224), as_strings=False,
                                                      print_per_layer_stat=False)
    print('calculation full finish')

    net_mask = tem_MobileNetV2(net_mask).cpu()
    net_mask.eval()

    pruneflops, pruneparams = get_model_complexity_info(net_mask, (3, 224, 224), as_strings=False,
                                                        print_per_layer_stat=False)

    print('calculation pruned finish')

    print('flops% = ', pruneflops/fullflops)
    print('parameters% = ', pruneparams/fullparams)
    return fullflops, pruneflops, fullparams, pruneparams