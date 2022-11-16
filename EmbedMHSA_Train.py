# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# In[1]:
from nnutils.all import *
from nnutils.resnets import *
from torch.optim import *
# import torchvision.models as models
# import torch.nn as nn

# Environment:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(torch.cuda.is_available(), device)
# Setting:
bs = 80#64
lr = 1e-3
first_train = True # False True
opt = 'Adamax' # Adamax SGD
cls_ft_channel = 256
sa_dim = 256
model_name = "TanhAdd1Div2_Flkr_LR{}".format(str(lr))
# model_name = "Map-Red-MHSA={}d_lr={}".format(str(sa_dim),str(lr))
logger = initLogger(model_name)

logger.info("Freeze ConvLayer, Train map, reduce, embed_weight(SC), MultiheadInteraction(no SC)and FC")
logger.info("Batch_Size ="+str(bs)+" Optimizer ="+ opt +" Lr ="+ str(lr))
# logger.info('cls_ft_channel='+str(cls_ft_channel))
# logger.info('sa_dim='+str(sa_dim))
root = 'D:/dataset/Sentiment_LDL/Flickr_LDL/images/'
# root = './Flickr_LDL/images/' # Ebd need modify, see ln 526
# root = './Twitter_LDL/images/'
logger.info("【Model Name】"+model_name)
logger.info("【Data root】"+root)
vote_num = 11
latest_model_path = './ckpt/{}_latest.pth'.format(model_name)
early_stopping_COS = EarlyStopping(name=model_name, patience=15)
early_stopping_KL = EarlyStopping(name=model_name, patience=15)
epsilon = 1e-6
EMD = EMDLoss() 
# L1Std = L1StdLoss()
from tensorboardX import SummaryWriter
writerKL = SummaryWriter('./stat/{}/KL'.format(model_name), comment = model_name)
writerCOS = SummaryWriter('./stat/{}/COS'.format(model_name), comment = model_name)
writerAcc = SummaryWriter('./stat/{}/Acc'.format(model_name), comment = model_name)


# %
# 网络组建
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BranchReduce(nn.Module):
    def __init__(self, channel):
        super(BranchReduce, self).__init__()
        self.channel = channel
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(channel, 1)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
# import torch.nn as nn
class MultiheadInteraction(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadInteraction, self).__init__()
        self.d_model = d_model
        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self._qkv_same_embed_dim = self.kdim == d_model and self.vdim == d_model

        self.num_heads = num_heads
        self.dropout = dropout
        self.depth = d_model // num_heads
        assert self.depth * num_heads == self.d_model, "embed_dim must be divisible by num_heads"

    def myAct(self, x):
        func = nn.Tanh()
        return (func(x) + 1)/2

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads,depth)
        Arguments:
        x -- A tokenized sequence (batch_size,seq_len,d_model)
        Returns:
        A tokenized sequence with dimensions (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x

    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)
        # self.act2 self.act1
        q = self.myAct(self.split_heads(q, batch_size))  # (batch_size, seq_len_q, num_heads, depth_q) (m,l,h,d)
        k = self.myAct(self.split_heads(k, batch_size))  # (batch_size,  seq_len_k, num_heads, depth_q) (m,j,h,d)
        v = self.split_heads(v, batch_size)  # (batch_size,  seq_len_v, num_heads, depth_v) (m,j,h,e)

        kv = torch.einsum("mjhd,mjhe->mdeh", k, v)  # (batch_size, depth_k, depth_v, seq_len_v)

        k_reduced = torch.sum(k, dim=1) + 1e-8

        z = 1 / (torch.einsum("mlhd,mhd->mlh", q, k_reduced))  # (batch_size, num_heads, seq_len_q)

        output = torch.einsum("mlhd,mdeh,mlh->mlhe", q, kv, z)  # (batch_size,len_q, heads, depth_v)
        output = torch.reshape(output, (batch_size, -1, self.num_heads * self.depth))  # (batch_size,len_q, d_model)

        return output

import torch.nn as nn

class FeatEmbedFusion(nn.Module):
    def __init__(self, feat_channel, embed_dim):
        super(FeatEmbedFusion, self).__init__()
        self.fc1 = nn.Linear(embed_dim, feat_channel+embed_dim)
        self.fc2 = nn.Linear(feat_channel+embed_dim, feat_channel)
        self.relu = nn.ReLU()
        self.sf = nn.Softmax(dim=-1)
    def forward(self, feat, embed):
        x = self.relu(self.fc1(embed))
        x = self.sf(self.fc2(x))
        x = feat + x
        return x

class InnerAttention(nn.Module):
    '''
    the last dim will be changed to sa_dim
    '''
    def __init__(self, embed_dim=256, sa_dim=256, num_heads=8):
        super(InnerAttention, self).__init__()
        # self.scReduce = nn.Linear(embed_dim, sa_dim)
        self.Q_layer = nn.Linear(embed_dim, sa_dim)
        self.K_layer = nn.Linear(embed_dim, sa_dim)
        self.V_layer = nn.Linear(embed_dim, sa_dim)
        self.mfi_layer = MultiheadInteraction(d_model=sa_dim, num_heads=num_heads)
        # self.mha_layer = nn.MultiheadAttention(embed_dim=sa_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # print("\n Entry InnerAttention:",x.shape)
        # x = x.permute([1,0,2])
        q = self.Q_layer(x)
        k = self.K_layer(x)
        v = self.V_layer(x)
        out = self.mfi_layer(q,k,v)
        # out, _ = self.mha_layer(q,k,v)
        # print("Exit mha_layer:",x.shape)
        x = self.dropout(out)
        # x = x.permute([1,0,2])
        # print("InnerAttention out:",x.shape)
        return x

class LayerOutput(nn.Module):
    def __init__(self, channel):
        super(LayerOutput, self).__init__()
        self.channel = channel
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel, 8)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
# %%
# 框架内容
class ResNet(nn.Module):
    def __init__(self, block, layers, embeds, num_classes=8, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cls_ft_channel=256, sa_dim=256):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # 每类的feature参与关系推理时使用的channel数：
        self.cls_ft_channel = cls_ft_channel
        # 每类Reduce之后的特征维度
        self.class_feature = 256
        self.sa_dim = sa_dim
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer_map = nn.ModuleList([self._Map2ClsBranch() for i in range(8)])
        self.layer_reduce = nn.ModuleList([BranchReduce(self.cls_ft_channel) for i in range(8)])
        self.layer_MHA = InnerAttention(embed_dim=self.cls_ft_channel, sa_dim=self.sa_dim, num_heads=8)

        self.layer_emb = nn.ModuleList([FeatEmbedFusion(self.cls_ft_channel, embed_dim=50) for i in range(8)])

        self.embeds = embeds

        # self.output = LayerOutput(1024)
        self.fc = nn.Linear(8*self.sa_dim,8)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _Map2ClsBranch(self):
        return nn.Sequential(
                    nn.Conv2d(2048,self.cls_ft_channel, 3),
                    nn.BatchNorm2d(self.cls_ft_channel),
                    self.relu
                )
    # def _make_NL_branch(self):
    #     return nn.Sequential(
    #                 NLBlock(self.cls_ft_channel),
    #             )

    def _make_downdim(self):
        return nn.Sequential(
                    torch.nn.Conv2d(self.cls_ft_channel*2, self.cls_ft_channel, 3),
                    nn.BatchNorm2d(self.cls_ft_channel),
                    self.relu
                )
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # print("layer1:", x.shape)

        x = self.layer2(x)
        # print("layer1:", x.shape)
        
        x = self.layer3(x)
        # print("layer3:", x.shape)

        x = self.layer4(x) # 512,2048,7,7
        # # 2048 Map to 8 branches(cls_ft_channel = 128)
        
        x_mapped_list = [self.layer_map[i](x) for i in range(8)]

        x_reduced_list = [self.layer_reduce[i](x_mapped_list[i]) for i in range(8)] # mapped->reduced

        x_enhanced_list = [self.layer_emb[i](x_reduced_list[i], self.embeds[i]) for i in range(8)]

        # ↑ 512*300 = bs*cls_ft_channel
        
        y = torch.stack(x_enhanced_list, 1) # 512,8,300 = bs, cls_num, feat_dim
        # print("Entry layer_MHA:",y.shape)

        y_ = self.layer_MHA(y)
        # print("layer_MHA(y):", y_.shape)

        y = y_.flatten(1)

        y = self.fc(y)
        # print("fc out:", y.shape)
        y = self.sigmoid(y)
        # x = self.output(y)

        return y

    def forward(self, x):
        return self._forward_impl(x)

num_classes = 8
embeds_Twitter = torch.tensor([[-1.2454e-01,  7.1525e-01, -3.0322e-01,  9.6026e-01, -4.9850e-01,
          1.5668e+00,  1.0234e+00, -1.4680e+00, -8.4014e-01,  4.1812e-01,
          8.2496e-01, -6.1171e-01, -3.3037e+00, -6.0072e-01,  4.8982e-01,
         -1.5486e-01,  1.8391e-01, -2.0090e-02,  5.9786e-01, -1.1447e+00,
         -3.9170e-01,  5.9087e-01, -8.4199e-02,  7.5692e-01,  1.7650e-01,
          5.6192e-01,  8.6024e-02,  1.0297e+00, -8.6369e-01,  7.2424e-01,
          8.3881e-02, -1.0526e+00, -6.8373e-01,  1.2508e-01, -1.5011e-01,
          2.5301e-01, -7.3741e-01,  1.7974e-01,  1.4765e-01,  4.9873e-02,
          7.8587e-01,  2.2052e-01,  5.1943e-01, -1.9269e-01,  1.1030e-01,
          1.8299e-01,  2.7745e-01, -7.8577e-02,  5.2635e-01,  5.8689e-01],
        [-1.0706e+00,  3.9100e-01,  2.3514e-02,  9.9357e-01, -5.0482e-01,
          7.9003e-01,  1.2922e+00, -1.4342e+00, -1.2091e+00,  1.7843e-01,
          1.2244e+00, -9.5288e-01, -2.7226e+00,  1.4098e-01, -5.1142e-01,
          4.0862e-01,  4.3471e-01,  7.4527e-01,  3.8697e-01, -1.3532e+00,
         -1.3879e-01,  3.6309e-01, -3.3709e-01,  1.0214e+00,  4.3910e-01,
          3.9724e-01,  1.3766e-01,  5.9264e-01, -5.8806e-02,  1.4484e-01,
         -9.1732e-02, -1.3277e+00, -6.6579e-01,  5.3713e-01, -5.3118e-01,
         -2.0395e-01, -4.5697e-01, -5.4050e-01, -1.9972e-01,  7.4334e-01,
          5.9521e-01, -5.3063e-01,  2.3529e-01, -4.7135e-01, -4.3485e-01,
          4.6704e-01,  1.2553e-01,  2.7016e-01,  7.9702e-02,  4.7887e-01],
        [ 2.9973e-01,  7.9273e-01,  6.8428e-01,  3.7039e-01, -7.7391e-01,
         -1.8345e-01,  4.3283e-01, -4.4464e-02, -1.8822e-01,  5.5892e-01,
         -7.2734e-01,  7.2173e-01, -2.3364e+00, -5.3720e-01,  7.5460e-02,
          4.7451e-01,  5.1545e-01, -1.9671e-01, -1.2829e+00,  1.2070e-01,
          2.8088e-01, -6.0573e-02,  2.9848e-01,  8.2943e-01, -6.9705e-01,
          2.6957e-01, -6.4852e-01,  1.5678e-01,  1.0276e+00, -6.0336e-01,
         -5.7244e-01,  4.7162e-01,  1.4614e-01, -7.3956e-02, -5.6521e-01,
          2.9580e-01, -1.4999e-01,  3.8320e-01, -3.3260e-01,  2.7323e-01,
         -2.9537e-01, -2.1855e-01,  7.6762e-01,  6.9457e-01, -1.1000e+00,
          1.7480e-01,  6.4274e-01, -1.1164e-01,  2.2044e-01,  3.8133e-01],
        [-5.1244e-04,  6.1110e-01,  6.6274e-02,  7.7193e-01, -5.2385e-01,
          9.5209e-01,  6.6223e-01, -6.3778e-02, -5.8343e-01,  4.8529e-01,
          9.1274e-01, -3.2100e-01, -4.0315e+00,  2.6784e-01, -3.6299e-01,
          1.3601e-01,  3.6549e-01,  7.3048e-01,  1.2236e+00, -1.1892e+00,
         -1.4966e-01,  1.5890e-01, -9.4793e-02,  7.5576e-01,  4.0128e-01,
          9.2919e-01, -2.3436e-01,  4.1686e-01, -3.9044e-02,  4.2702e-01,
          2.8576e-01, -1.1523e+00,  1.5738e-01,  5.4279e-01, -2.0122e-01,
          1.8790e-01, -5.6922e-01, -4.3987e-01, -9.2860e-02, -6.1148e-01,
          2.8811e-01, -2.7582e-01,  4.2850e-01, -6.6044e-01,  3.0342e-01,
          1.7036e-01,  3.3291e-01,  6.1700e-01,  8.3661e-01,  3.4058e-01],
        [ 6.8282e-01,  4.1474e-01,  8.0976e-01,  4.2422e-01,  1.1849e-01,
          1.8743e-01,  3.9879e-01, -8.4030e-01, -8.4501e-01,  7.3312e-01,
          9.5993e-01, -4.1664e-01, -1.7842e+00, -9.4098e-01,  3.7277e-01,
          4.9862e-01, -1.3767e-01, -9.0746e-01,  3.8757e-01, -3.3572e-01,
         -3.3319e-03, -6.2475e-01,  3.3128e-01,  8.3693e-01,  8.3627e-02,
          1.2030e+00,  1.8875e-01,  1.0537e+00, -8.0380e-01,  2.5073e-01,
         -4.2631e-01, -3.0053e-01,  1.8594e-01, -1.2491e-01, -9.4514e-01,
         -4.7293e-01, -9.1991e-01, -5.9154e-02, -1.6086e-01, -2.2215e-02,
          1.6236e+00,  4.7306e-02,  1.6578e-01, -3.8505e-01, -2.0728e-01,
          1.1620e-01,  1.5143e+00,  4.0899e-01,  3.8078e-01,  8.5031e-02],
        [-2.2608e-01,  4.9054e-01,  6.6832e-01,  2.2291e-03, -3.0691e-01,
          1.1035e+00,  1.2723e+00, -1.5366e+00, -3.8758e-01, -8.2419e-02,
          8.2507e-01, -6.8070e-01, -2.8531e+00,  3.6523e-01, -4.9950e-01,
          9.0850e-01,  4.1824e-01,  1.1603e-01, -4.2875e-01, -4.2609e-01,
          3.5029e-03,  2.4624e-01,  3.8781e-01,  1.2361e+00, -5.0160e-01,
          1.0297e+00, -3.4882e-01,  2.9896e-01, -4.3197e-01, -3.4412e-02,
         -1.8895e-01, -1.0857e+00, -1.0516e-01,  1.0552e-01,  2.4564e-01,
         -1.2962e-01, -2.0569e-01, -2.3998e-01,  1.5097e-01,  6.7582e-01,
          7.9723e-01, -3.5419e-01,  3.8002e-01,  2.7344e-01, -3.3005e-02,
          1.0295e+00,  6.2739e-02, -1.9345e-01, -2.2440e-01,  1.2787e+00],
        [-3.6268e-01,  2.6175e-01, -5.3587e-01,  3.4502e-02,  6.4340e-01,
         -6.9381e-02,  6.8583e-01, -1.1833e+00, -1.4565e-01, -5.2321e-01,
          1.8037e-01,  1.4385e-01, -1.4491e+00,  3.0350e-01,  7.3463e-01,
         -1.0620e-01,  4.9953e-01, -2.4501e-01, -4.9000e-01, -2.5306e-01,
          1.1210e+00, -5.2576e-01,  2.8440e-01, -1.3252e-01, -1.1538e-01,
          7.0070e-01, -2.9650e-01,  5.9392e-01,  1.0850e-01, -8.0077e-01,
          7.8099e-01, -5.9053e-01,  1.6978e-01, -6.7163e-01, -2.8046e-01,
         -1.0582e-01,  1.0775e-01, -2.3658e-01,  7.9799e-01, -2.9181e-01,
          4.0502e-01,  4.3996e-01, -2.5999e-01, -5.3395e-01,  2.5970e-02,
          1.2648e+00,  4.7779e-01,  1.5795e-01,  3.3921e-01,  3.1521e-01],
        [-1.1407e+00, -3.7904e-01, -4.7108e-01,  1.2201e+00,  1.1387e-01,
          1.9751e+00, -6.2628e-01, -1.2268e+00, -8.4925e-01, -6.8875e-01,
          4.5420e-01, -1.8537e-01, -1.5264e+00, -4.6070e-01,  3.8199e-01,
          7.8405e-01, -1.0988e-02,  8.3620e-01,  1.0795e+00, -9.6112e-02,
         -3.6707e-01,  7.8461e-01, -9.3250e-01,  9.9355e-01, -1.3808e-01,
          1.7160e+00,  3.3356e-01, -3.8712e-01,  1.2818e+00, -1.0745e+00,
          1.0397e+00, -8.8965e-01, -1.2139e-01, -2.6336e-01,  3.0496e-01,
         -6.8769e-01, -1.0898e-02,  7.1647e-02, -3.2448e-01, -4.7875e-01,
          4.0684e-01, -7.0148e-01,  5.8235e-01, -7.5386e-01, -2.5203e-01,
          2.1821e-02,  2.4996e-01,  5.3341e-01,  5.9072e-01,  2.8787e-01]])

embeds_Flickr = torch.tensor([[-3.6268e-01,  2.6175e-01, -5.3587e-01,  3.4502e-02,  6.4340e-01,
         -6.9381e-02,  6.8583e-01, -1.1833e+00, -1.4565e-01, -5.2321e-01,
          1.8037e-01,  1.4385e-01, -1.4491e+00,  3.0350e-01,  7.3463e-01,
         -1.0620e-01,  4.9953e-01, -2.4501e-01, -4.9000e-01, -2.5306e-01,
          1.1210e+00, -5.2576e-01,  2.8440e-01, -1.3252e-01, -1.1538e-01,
          7.0070e-01, -2.9650e-01,  5.9392e-01,  1.0850e-01, -8.0077e-01,
          7.8099e-01, -5.9053e-01,  1.6978e-01, -6.7163e-01, -2.8046e-01,
         -1.0582e-01,  1.0775e-01, -2.3658e-01,  7.9799e-01, -2.9181e-01,
          4.0502e-01,  4.3996e-01, -2.5999e-01, -5.3395e-01,  2.5970e-02,
          1.2648e+00,  4.7779e-01,  1.5795e-01,  3.3921e-01,  3.1521e-01],
        [-1.2454e-01,  7.1525e-01, -3.0322e-01,  9.6026e-01, -4.9850e-01,
          1.5668e+00,  1.0234e+00, -1.4680e+00, -8.4014e-01,  4.1812e-01,
          8.2496e-01, -6.1171e-01, -3.3037e+00, -6.0072e-01,  4.8982e-01,
         -1.5486e-01,  1.8391e-01, -2.0090e-02,  5.9786e-01, -1.1447e+00,
         -3.9170e-01,  5.9087e-01, -8.4199e-02,  7.5692e-01,  1.7650e-01,
          5.6192e-01,  8.6024e-02,  1.0297e+00, -8.6369e-01,  7.2424e-01,
          8.3881e-02, -1.0526e+00, -6.8373e-01,  1.2508e-01, -1.5011e-01,
          2.5301e-01, -7.3741e-01,  1.7974e-01,  1.4765e-01,  4.9873e-02,
          7.8587e-01,  2.2052e-01,  5.1943e-01, -1.9269e-01,  1.1030e-01,
          1.8299e-01,  2.7745e-01, -7.8577e-02,  5.2635e-01,  5.8689e-01],
        [ 2.9973e-01,  7.9273e-01,  6.8428e-01,  3.7039e-01, -7.7391e-01,
         -1.8345e-01,  4.3283e-01, -4.4464e-02, -1.8822e-01,  5.5892e-01,
         -7.2734e-01,  7.2173e-01, -2.3364e+00, -5.3720e-01,  7.5460e-02,
          4.7451e-01,  5.1545e-01, -1.9671e-01, -1.2829e+00,  1.2070e-01,
          2.8088e-01, -6.0573e-02,  2.9848e-01,  8.2943e-01, -6.9705e-01,
          2.6957e-01, -6.4852e-01,  1.5678e-01,  1.0276e+00, -6.0336e-01,
         -5.7244e-01,  4.7162e-01,  1.4614e-01, -7.3956e-02, -5.6521e-01,
          2.9580e-01, -1.4999e-01,  3.8320e-01, -3.3260e-01,  2.7323e-01,
         -2.9537e-01, -2.1855e-01,  7.6762e-01,  6.9457e-01, -1.1000e+00,
          1.7480e-01,  6.4274e-01, -1.1164e-01,  2.2044e-01,  3.8133e-01],
        [-1.1407e+00, -3.7904e-01, -4.7108e-01,  1.2201e+00,  1.1387e-01,
          1.9751e+00, -6.2628e-01, -1.2268e+00, -8.4925e-01, -6.8875e-01,
          4.5420e-01, -1.8537e-01, -1.5264e+00, -4.6070e-01,  3.8199e-01,
          7.8405e-01, -1.0988e-02,  8.3620e-01,  1.0795e+00, -9.6112e-02,
         -3.6707e-01,  7.8461e-01, -9.3250e-01,  9.9355e-01, -1.3808e-01,
          1.7160e+00,  3.3356e-01, -3.8712e-01,  1.2818e+00, -1.0745e+00,
          1.0397e+00, -8.8965e-01, -1.2139e-01, -2.6336e-01,  3.0496e-01,
         -6.8769e-01, -1.0898e-02,  7.1647e-02, -3.2448e-01, -4.7875e-01,
          4.0684e-01, -7.0148e-01,  5.8235e-01, -7.5386e-01, -2.5203e-01,
          2.1821e-02,  2.4996e-01,  5.3341e-01,  5.9072e-01,  2.8787e-01],
        [ 6.8282e-01,  4.1474e-01,  8.0976e-01,  4.2422e-01,  1.1849e-01,
          1.8743e-01,  3.9879e-01, -8.4030e-01, -8.4501e-01,  7.3312e-01,
          9.5993e-01, -4.1664e-01, -1.7842e+00, -9.4098e-01,  3.7277e-01,
          4.9862e-01, -1.3767e-01, -9.0746e-01,  3.8757e-01, -3.3572e-01,
         -3.3319e-03, -6.2475e-01,  3.3128e-01,  8.3693e-01,  8.3627e-02,
          1.2030e+00,  1.8875e-01,  1.0537e+00, -8.0380e-01,  2.5073e-01,
         -4.2631e-01, -3.0053e-01,  1.8594e-01, -1.2491e-01, -9.4514e-01,
         -4.7293e-01, -9.1991e-01, -5.9154e-02, -1.6086e-01, -2.2215e-02,
          1.6236e+00,  4.7306e-02,  1.6578e-01, -3.8505e-01, -2.0728e-01,
          1.1620e-01,  1.5143e+00,  4.0899e-01,  3.8078e-01,  8.5031e-02],
        [-2.2608e-01,  4.9054e-01,  6.6832e-01,  2.2291e-03, -3.0691e-01,
          1.1035e+00,  1.2723e+00, -1.5366e+00, -3.8758e-01, -8.2419e-02,
          8.2507e-01, -6.8070e-01, -2.8531e+00,  3.6523e-01, -4.9950e-01,
          9.0850e-01,  4.1824e-01,  1.1603e-01, -4.2875e-01, -4.2609e-01,
          3.5029e-03,  2.4624e-01,  3.8781e-01,  1.2361e+00, -5.0160e-01,
          1.0297e+00, -3.4882e-01,  2.9896e-01, -4.3197e-01, -3.4412e-02,
         -1.8895e-01, -1.0857e+00, -1.0516e-01,  1.0552e-01,  2.4564e-01,
         -1.2962e-01, -2.0569e-01, -2.3998e-01,  1.5097e-01,  6.7582e-01,
          7.9723e-01, -3.5419e-01,  3.8002e-01,  2.7344e-01, -3.3005e-02,
          1.0295e+00,  6.2739e-02, -1.9345e-01, -2.2440e-01,  1.2787e+00],
        [-5.1244e-04,  6.1110e-01,  6.6274e-02,  7.7193e-01, -5.2385e-01,
          9.5209e-01,  6.6223e-01, -6.3778e-02, -5.8343e-01,  4.8529e-01,
          9.1274e-01, -3.2100e-01, -4.0315e+00,  2.6784e-01, -3.6299e-01,
          1.3601e-01,  3.6549e-01,  7.3048e-01,  1.2236e+00, -1.1892e+00,
         -1.4966e-01,  1.5890e-01, -9.4793e-02,  7.5576e-01,  4.0128e-01,
          9.2919e-01, -2.3436e-01,  4.1686e-01, -3.9044e-02,  4.2702e-01,
          2.8576e-01, -1.1523e+00,  1.5738e-01,  5.4279e-01, -2.0122e-01,
          1.8790e-01, -5.6922e-01, -4.3987e-01, -9.2860e-02, -6.1148e-01,
          2.8811e-01, -2.7582e-01,  4.2850e-01, -6.6044e-01,  3.0342e-01,
          1.7036e-01,  3.3291e-01,  6.1700e-01,  8.3661e-01,  3.4058e-01],
        [-1.0706e+00,  3.9100e-01,  2.3514e-02,  9.9357e-01, -5.0482e-01,
          7.9003e-01,  1.2922e+00, -1.4342e+00, -1.2091e+00,  1.7843e-01,
          1.2244e+00, -9.5288e-01, -2.7226e+00,  1.4098e-01, -5.1142e-01,
          4.0862e-01,  4.3471e-01,  7.4527e-01,  3.8697e-01, -1.3532e+00,
         -1.3879e-01,  3.6309e-01, -3.3709e-01,  1.0214e+00,  4.3910e-01,
          3.9724e-01,  1.3766e-01,  5.9264e-01, -5.8806e-02,  1.4484e-01,
         -9.1732e-02, -1.3277e+00, -6.6579e-01,  5.3713e-01, -5.3118e-01,
         -2.0395e-01, -4.5697e-01, -5.4050e-01, -1.9972e-01,  7.4334e-01,
          5.9521e-01, -5.3063e-01,  2.3529e-01, -4.7135e-01, -4.3485e-01,
          4.6704e-01,  1.2553e-01,  2.7016e-01,  7.9702e-02,  4.7887e-01]])
        
model = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group = 64 * 2, 
cls_ft_channel=cls_ft_channel, sa_dim=sa_dim, embeds = embeds_Flickr.cuda()).cuda()
# print(model)

# %%
model = nn.DataParallel(model)
torch.backends.cudnn.benchmark = True
# 原始网络参数

if first_train:
    epoch_start = 0
    ckpt_path ='./ckpt/WRS-EMD_only_Adamax_4_test0.42917.pth'
    # ckpt_path ='./ckpt/Base_Twitter_lr_0.0001_Ep48_test0.85903.pth'
    logger.info("First_train\nLoaded CKPT:"+ ckpt_path)
    wrn_net_dict = (torch.load(ckpt_path))
    dict_trained = wrn_net_dict
    dict_new = model.state_dict().copy()
    new_list = list ( model.state_dict().keys() )
    trained_list = list ( dict_trained.keys() )
    
    # 6-65为Layer1；66-143为Layer2；144-257为Layer3；258-317为Layer4
    for i in range(318): 
        dict_new[ new_list[i] ] = dict_trained[ trained_list[i] ]
    logger.info("Only loaded partial params.")
    model.load_state_dict(dict_new)
else:
    epoch_start = 1

    latest_model_path ='./ckpt/TanhAdd1Div2_Flkr_LR0.001_latest.pth'
    # latest_model_path ='./ckpt/EmbedWeight_MHSA_onFlickr_LR0.001_Ep22_test0.83799.pth'
    loaded_dict = torch.load(latest_model_path)
    # trained_list = list ( loaded_dict.keys() )# len=376, 最后两个为FC的参数，可删
    # print("dict.keys():\n", trained_list)
    model.load_state_dict(loaded_dict)
    logger.info("Success_train\nLoaded CKPT:"+ latest_model_path)


# %%
# # 冻结读入参数的层
i = 0
for child in model.children():
    # print(child, type(child),'\n\n')
    for layer in child.children():
        
        # 冻结前i层
        for param in layer.parameters():
            param.requires_grad = False
        # 第i层冻完了，退出 训练第i+1层
        if i == 7: # 0-3层不是Block; 4,5,6,7才是Layer1,2,3,4
            print("The last frozen layer is {} ".format(i-3), layer)
            break
        i += 1
    break

# 排除被冻结层参数
opt_params = filter(lambda p: p.requires_grad, model.parameters())
# 优化器
import torch
if opt=='Adamax':
    # optimizer = torch.optim.Adamax(opt_params, lr=lr)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
elif opt=='SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# 余弦衰减
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30, eta_min = 1e-4)

# %%
train_dataset = LDLdataset('train_processed.txt',mode="train", root=root,  vote_num = vote_num)
test_dataset = LDLdataset('test_processed.txt',mode="test", root=root,  vote_num = vote_num)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
len(train_dataset),len(test_dataset)

def train():
    # scheduler.step()
    model.train()
    loss_all = 0
    num_data = 1
    for data, ground, fn in train_loader:
#     for data, cls, fn in eval_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = ground.to(device)
        # l1 = L1Std((output),(label)) L1与EMD两种loss没有相差很悬殊(3跟2)
        # Loss_Function
        loss = EMD((output),(label))# + L1Std((output),(label))*0.1

        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        if num_data==1 or num_data%200 == 0:
            print("【train】", num_data, time.strftime( "%H:%M",time.localtime() )  )
        num_data += 1
    return loss_all / len(train_dataset)


# %%

def evaluate(loader,name):
    # global model
    model.eval()
    total = 0
    top1 = 0
    predictions = []
    labels = []
    kl_all = 0
    # emd_all = 0
    cheb_all = 0
    itsc_all = 0
    cos_all = 0
    clark_all = 0
    canber_all = 0
    global epsilon #1e-6
    with torch.no_grad():
        for data, ground, fn in tqdm(loader):
            
            data = data.to(device)

            output = model(data)
            ground = ground.detach().cpu()
            pred = (output).detach().cpu()
            t_data_x = pred

            a = (pred)
            b = (ground)
            
            kl_all += KL_sum((a + epsilon).log(), b).item()

            # emd_all += EMD(a, b).item()

            cheb_dis = abs(a-b).max(dim=1).values.sum().item()
            cheb_all += cheb_dis

            # mask_a = torch.le(a, b)
            # mask_b = ~mask_a
            # itsc_dis = (a.mul(mask_a) + b.mul(mask_b)).sum().item()
            itsc_dis = torch.min(a, b).sum().item()
            itsc_all += itsc_dis
                        
            clark_all += ((a-b).pow(2)/((a+b).pow(2) +epsilon)).sum(dim=1).pow(1/2).sum().item()
            canber_all += (abs(a-b)/(a+b +epsilon)).sum().item()
            cos_all += COS((pred),(ground)).sum().item()
            
            # pred = torch.max(pred,1)[1] 
            # label = torch.max(ground,1)[1] .detach().cpu()
            # Acc：
            row = 0
            max_val_list = []
            max_val_pos = torch.max(pred,1)[1]
            gt_values = torch.max(ground,1)[0]
            for col in max_val_pos:
                max_val_list.append(ground[row,col.item()])
                row+=1
            max_vals = torch.stack(max_val_list, 0)
            correct_nums = (max_vals == gt_values).sum().item()
            top1 += correct_nums
            # _,max2 = torch.topk(t_data_x,2,dim=-1) # origin pred

            # # total += label.size(0)
            # label = label.view(-1,1)
            # top1 += (label == max2[:,0:1]).sum().item()
            
    # predictions = np.hstack(predictions)
    # labels = np.hstack(labels)
    total = len(loader.dataset)
    Acc = top1/total
    meanKL = kl_all/total
    # meanEDM = emd_all/total
    meanCOS = cos_all/total
    meanCheb = cheb_all/total
    meanItsc = itsc_all/total
    meanClark = clark_all/total
    meanCanber = canber_all/total
#     print("predictions,labels",predictions,labels)
    # print("【Eval_" + name + "】topK Acc:", Acc,  time.strftime( "%H:%M",time.localtime() ))
    logger.info("an epoch evaluated：\nAcc = %s, Cheb = %s, Clark = %s, Canber = %s, ", str(Acc), str(meanCheb), str(meanClark), str(meanCanber))
    logger.info("KLdiv = %s, Cosine = %s, Itsc = %s", str(meanKL), str(meanCOS), str(meanItsc))
    #, str(meanItsc))

    # print("Cheb:(<0.25)",meanCheb)
    # print("Clark:(<2.2)",meanClark)
    # print("Canber:(<5.5)",meanCanber)
    # print("KLdiv:(<0.45)",meanKL)
    # print("Cosine:(>0.84)",meanCOS)
    # print("Itsc:(>0.6)",meanItsc)

    return top1/total, meanCOS, meanKL
    
# %%
# 提前测试网络
test_acc, test_COS, test_KL = evaluate(test_loader,"test_loader")
print(test_acc, test_COS, test_KL)

# writerKL.add_scalar(model_name, test_KL, global_step=epoch_start)
# writerCOS.add_scalar(model_name, test_COS, global_step=epoch_start)
# writerAcc.add_scalar(model_name, test_acc, global_step=epoch_start)
# logger.info("Before train：test_acc = %s, test_COS = %s, test_KL = %s",str(test_acc),str(test_COS),str(test_KL))

# %%
# 开始训练
for e in range(epoch_start,100):
    epoch_start += 1
    logger.info("an epoch started: %s", str(epoch_start))
    loss = train()
    torch.save(model.state_dict(), latest_model_path)
    logger.info("an epoch trained：loss = %s", str(loss))

    # train_acc, train_COS, train_KL = evaluate(train_loader,"train_loader")
    test_acc, test_COS, test_KL = evaluate(test_loader,"test_loader")
    
    writerKL.add_scalar(model_name, test_KL, global_step=e)
    writerCOS.add_scalar(model_name, test_COS, global_step=e)
    writerAcc.add_scalar(model_name, test_acc, global_step=e)

    logger.info("an epoch evaluated：\ntest_acc = %s, test_COS = %s, test_KL = %s",str(test_acc),str(test_COS),str(test_KL))
    print('Epoch: {:03d}, Loss: {:.5f} , Test Auc: {:.5f}, \nTest COS: {:.5f}, Test KL: {:.5f}'.
          format(epoch_start, loss, test_acc, test_COS, test_KL))
    
    if epoch_start>5: early_stopping_KL(epoch_start, test_KL, model, tend='inverse')
    early_stopping_COS(epoch_start, test_COS, model, tend='direct')
    if early_stopping_COS.early_stop:
        print("Early stopping")
        logger.info("Early stopping")
        break
        
    


# %%


        # # Relation Inference, channel not be changed
        # x_infered_list = []
        # for i in range(8):
        #     group_out_list = []
        #     # Combination of atted_feat i and j (from cls_ft_channel to cls_ft_channel*2)
        #     for j in range(8):
        #         if i==j : continue
        #         group_in = torch.cat([x_attded_list[j], x_attded_list[i]], 1) 
        #         # 1x1 Conv make dim down (from cls_ft_channel*2 to cls_ft_channel)
        #         group_out = self.layer_downdim[i](group_in) 
        #         group_out_list.append(group_out)
        #     # Sum of Comb_ij
        #     tensor_sum = torch.ones_like(group_out_list[0])
        #     for tensor_item in group_out_list:
        #         tensor_sum += tensor_item
        #     # return rel_infered_i
        #     x_infered_list.append(tensor_sum)
        # x_reduced_list = [self.layer_reduce[i](x_infered_list[i]) for i in range(8)]

