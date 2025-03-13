from torch.nn import functional as F
from torch.nn import (
    Module,
    Sequential,
    ReLU,
    Hardswish,
    Conv2d,
    BatchNorm2d,
    Dropout,
    Linear,
    Hardsigmoid,
)


BN_EPS = 0.001
BN_MOMENTUM = 0.01
SE_REDUCTION = 4


class SqueezeExcite(Module):
    def __init__(self, expand_channels, reduction=SE_REDUCTION) -> None:
        super().__init__()

        self.fcR = Sequential(
            Linear(expand_channels, expand_channels // reduction, bias=False),
            ReLU(),
        )

        self.fcHS = Sequential(
            Linear(expand_channels // reduction, expand_channels, bias=False),
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

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            raise ValueError("Invalid kernel size, 3 and 5 only supported")

        self.conv1 = Sequential(
            Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(expand_channels, eps=bn_eps, momentum=bn_momentum),
            activation,
        )

        self.dconv1 = Sequential(
            Conv2d(
                expand_channels,
                expand_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=expand_channels,
                bias=False,
            ),
            BatchNorm2d(expand_channels, eps=bn_eps, momentum=bn_momentum),
        )

        self.squeeze = SqueezeExcite(expand_channels)

        self.conv2 = Sequential(
            Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum),
            activation,
        )

        self.connectFlag = in_channels == out_channels and stride == 1

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.dconv1(x)
        if self.use_se:
            x = self.squeeze(x)
        x = self.conv2(x)

        if self.connectFlag:
            x = x + identity

        return x


class CustomMobileNetSmall(Module):
    def __init__(
        self,
        n_classes,
        dropout=0.2,
        bn_eps=BN_EPS,
        bn_momentum=BN_MOMENTUM,
        se_reduction=SE_REDUCTION,
    ) -> None:
        super(CustomMobileNetSmall, self).__init__()
        self.n_classes = n_classes

        self.pConv1 = Sequential(
            Conv2d(3, 16, kernel_size=3, stride=2),
            BatchNorm2d(16, eps=bn_eps, momentum=bn_momentum),
            Hardswish(),
        )

        self.invRes = Sequential()
        # n-in-channels, n-out-channels, n-expand-channels, kernel-size, activation, use-se, stride
        layers = [
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
        for inp, out, exp, k, act, se, s in layers:
            self.invRes.append(
                Bottleneck(inp, out, exp, k, act, se, s, bn_eps, bn_momentum)
            )

        self.pConv2 = Sequential(
            Conv2d(96, 576, kernel_size=1, stride=1),
            SqueezeExcite(576, se_reduction),
            BatchNorm2d(576, eps=bn_eps, momentum=bn_momentum),
            Hardswish(),
        )

        self.pConv3 = Sequential(
            Conv2d(576, 1024, kernel_size=1, stride=1),
            Hardswish(),
            Dropout(dropout),
            Conv2d(1024, n_classes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.pConv1(x)
        x = self.invRes(x)
        x = self.pConv2(x)

        batch, _, h, w = x.size()
        x = F.avg_pool2d(x, (h, w))

        x = self.pConv3(x)
        x = x.view(batch, -1)
        return x
