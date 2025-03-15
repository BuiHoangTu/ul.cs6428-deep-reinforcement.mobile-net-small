import torch
from torch.nn import functional as F
from torch.nn import (
    Module,
    Sequential,
    ReLU,
    Hardswish,
    Conv2d,
    BatchNorm2d,
    Linear,
    Hardsigmoid,
    Flatten,
)


BN_EPS = 0.001
BN_MOMENTUM = 0.01
SE_REDUCTION = 4


def createConv2dBlock(
    in_channels,
    out_channels,
    kernel_size,
    activation,
    stride=1,
    padding=0,
    groups=1,
    bias=False,
    bn_eps=BN_EPS,
    bn_momentum=BN_MOMENTUM,
):
    """Create a Conv2d block with optional BatchNorm and activation

    Args:
        in_channels (int): number of input channels
        out_channels (int): _description_
        kernel_size (int or tuple[int, int]): _description
        activation (_type_): _description_
        stride (int, optional): _description_. Defaults to 1.
        padding (int, optional): _description_. Defaults to 0.
        groups (int, optional): _description_. Defaults to 1.
        bias (bool, optional): _description_. Defaults to False.
        bn_eps (int, optional): eps of batch norm. Defaults to BN_EPS.
        bn_momentum (int, optional): momentumn of batch norm. Defaults to BN_MOMENTUM.

    Returns:
        Sequential: a sequential block of Conv2d, BatchNorm2d and activation
    """

    layer = Sequential()
    layer.append(
        Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
    )
    layer.append(BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum))
    if activation is not None:
        layer.append(activation)
    return layer


class SqueezeExcite(Module):
    def __init__(self, expand_channels, reduction=SE_REDUCTION) -> None:
        super().__init__()

        compressed_channels = expand_channels // reduction

        self.fcR = Sequential(
            Linear(expand_channels, compressed_channels, bias=False),
            ReLU(),
        )

        self.fcHS = Sequential(
            Linear(compressed_channels, expand_channels, bias=False),
            Hardsigmoid(),
        )

    def forward(self, x):
        identity = x
        batch, c, h, w = x.size()

        x = F.avg_pool2d(x, (h, w)).view(batch, -1)
        x = self.fcR(x)
        x = self.fcHS(x)
        x = x.view(batch, c, 1, 1)

        return x * identity


class Bottleneck(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_channels,
        kernel_size,
        activation,
        use_se,
        stride,
        bn_eps=BN_EPS,
        bn_momentum=BN_MOMENTUM,
    ) -> None:
        super().__init__()
        self.use_se = use_se
        self.connectFlag = in_channels == out_channels and stride == 1

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            raise ValueError("Invalid kernel size, 3 and 5 only supported")

        self.conv1 = createConv2dBlock(
            in_channels,
            expand_channels,
            kernel_size=1,
            stride=1,
            activation=activation,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

        # depthwise convolution
        self.dconv1 = createConv2dBlock(
            expand_channels,
            expand_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=expand_channels,  # depthwise
            activation=activation,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

        self.squeeze = SqueezeExcite(expand_channels)

        self.conv2 = createConv2dBlock(
            expand_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            activation=None,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.dconv1(x)
        if self.use_se:
            x = self.squeeze(x)
        x = self.conv2(x)

        if self.connectFlag is True:
            x = x + identity

        return x


class CustomMobileNetSmall(Module):
    # n-in-channels, n-out-channels, n-expand-channels, kernel-size, activation, use-se, stride
    BNECKs_OF_SMALL = [
        (16, 16, 16, 3, ReLU(), True, 2),
        (16, 24, 72, 3, ReLU(), False, 2),
        (24, 24, 88, 3, ReLU(), False, 1),
        (24, 40, 96, 5, Hardswish(), True, 2),
        (40, 40, 240, 5, Hardswish(), True, 1),
        (40, 40, 240, 5, Hardswish(), True, 1),
        (40, 48, 120, 5, Hardswish(), True, 1),
        (48, 48, 144, 5, Hardswish(), True, 1),
        (48, 96, 288, 5, Hardswish(), True, 2),
        (96, 96, 576, 5, Hardswish(), True, 1),
        (96, 96, 576, 5, Hardswish(), True, 1),
    ]

    def __init__(
        self,
        in_channels,
        n_classes,
        bnecks=None,
        dropout=0.2,
        bn_eps=BN_EPS,
        bn_momentum=BN_MOMENTUM,
        se_reduction=SE_REDUCTION,
    ) -> None:
        super(CustomMobileNetSmall, self).__init__()
        self.n_classes = n_classes
        self.dropout = dropout

        self.conv1 = createConv2dBlock(
            in_channels,
            16,
            kernel_size=3,
            stride=2,
            activation=Hardswish(),
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

        self.invRes = Sequential()
        layers = bnecks or self.BNECKs_OF_SMALL

        for inp, out, exp, k, act, se, s in layers:
            self.invRes.append(
                Bottleneck(inp, out, exp, k, act, se, s, bn_eps, bn_momentum)
            )

        self.eConv2 = Sequential(
            Conv2d(96, 576, kernel_size=1, stride=1),
            SqueezeExcite(576, se_reduction),
            BatchNorm2d(576, eps=bn_eps, momentum=bn_momentum),
            Hardswish(),
        )

        # self.conv3 = createConv2dBlock(
        #     576,
        #     1024,
        #     kernel_size=1,
        #     stride=1,
        #     activation=Hardswish(),
        #     use_bn=False,
        # )
        # This has the same effect since the input is Cx1x1
        self.conv3 = Sequential(
            Flatten(),
            Linear(576, 1024),
            Hardswish(),
        )

        # self.classifier = createConv2dBlock(
        #     1024,
        #     n_classes,
        #     kernel_size=1,
        #     stride=1,
        #     activation=None,
        #     use_bn=False,
        # )
        self.classifier = Linear(1024, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.invRes(x)
        x = self.eConv2(x)

        _, _, h, w = x.size()
        x = F.avg_pool2d(x, (h, w))

        x = self.conv3(x)

        x = F.dropout(x, self.dropout)

        x = self.classifier(x)

        return torch.flatten(x, 1)
