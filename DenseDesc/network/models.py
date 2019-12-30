from torch import nn
from DenseDesc.network import wideresnet
from DenseDesc.network.network_utils import l2_normalize


class Net4Conv3Pool128DimAvgRes(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv0=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
        )
        self.conv0_short=nn.Sequential(
            nn.Conv2d(3,32,2,2),
            nn.InstanceNorm2d(32),
        )

        self.conv1=nn.Sequential(
            nn.Conv2d(32,64,5,1,2),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
        )
        self.conv1_short=nn.Sequential(
            nn.Conv2d(32,64,2,2),
            nn.InstanceNorm2d(64),
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(64,128,5,1,2),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
        )
        self.conv2_short=nn.Sequential(
            nn.Conv2d(64,128,2,2),
            nn.InstanceNorm2d(128),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(128,128,5,1,2),
            nn.InstanceNorm2d(128)
        )

    def forward(self, x):
        x=self.conv0(x)+self.conv0_short(x)
        x=self.conv1(x)+self.conv1_short(x)
        x=self.conv2(x)+self.conv2_short(x)
        x=self.conv3(x)

        x=x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x=l2_normalize(x)
        return x


class L2_Norm_wrapper(nn.Module):
    def forward(self, x):
        x = l2_normalize(x)
        return x


def get_WResNet18(cfg):
    """
    return the WResNet18 layers for DenseDesc
    :param num_branch: how many blocks are got rid of from the network (back)
    :return:
    """
    model_res_net = wideresnet.resnet18(num_classes=365)
    if 'num_branch' not in cfg.keys():
        raise RuntimeError('The configuration file does not have the key `num_branch`.')
    model_list = list(model_res_net.children())[:-1 * cfg['num_branch'] - 3]
    model_list.append(L2_Norm_wrapper())
    base_net = nn.Sequential(*model_list)
    return base_net
