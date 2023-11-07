import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Union, List, Dict, Any, cast
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class ProjectionHead(nn.Module):
    def __init__(self, input_size=1024, nn_size=512, encoding_size=100):
        super(ProjectionHead, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, encoding_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class EvaClassifier(nn.Module):
    def __init__(self, input_size=1024, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class AccNet(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 5,
        classifier_input_size: int = 1,
        classifier_layer_size: int = 2048,
        init_weights: bool = True,
    ) -> None:
        super(AccNet, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = cnn1()

    def forward(self, x):
        feats = self.feature_extractor(x)

        return feats


class SSLNET(nn.Module):
    def __init__(
        self, output_size=2, input_size=1024, number_nn=1024, flatten_size=1024
    ):
        super(SSLNET, self).__init__()
        self.feature_extractor = cnn1()
        self.rotation_classifier = Classifier(
            input_size=flatten_size, output_size=output_size
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        rotation_y = self.rotation_classifier(feats)
        # axis_y = self.axis_classifier(feats)

        return rotation_y


def make_layers(
    cfg: List[Union[str, int]], batch_norm: bool = False
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    i = 0
    while i < len(cfg):
        v = cfg[i]
        if v == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            i += 1
            my_kernel_size = cfg[i]

            v = cast(int, v)
            my_kernel_size = cast(int, my_kernel_size)

            conv1d = nn.Conv1d(
                in_channels, v, kernel_size=my_kernel_size, padding=1
            )
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
        i += 1
    return nn.Sequential(*layers)


class ConvBNReLU(nn.Module):
    """Convolution + batch normalization + ReLU is a common trio"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    ):
        super(ConvBNReLU, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.main(x)


class CNN(nn.Module):
    """Typical CNN design with pyramid-like structure"""

    def __init__(self, in_channels=3, num_filters_init=3):
        super(CNN, self).__init__()

        self.layer1 = ConvBNReLU(
            in_channels, num_filters_init, 3, 1, 1, bias=False
        )  # 900 -> 225
        self.layer2 = ConvBNReLU(
            num_filters_init, num_filters_init, 3, 1, 1, bias=False
        )  # 225 -> 56

        self.layer3 = ConvBNReLU(
            num_filters_init, num_filters_init * 2, 3, 1, 1, bias=False
        )  # 56 -> 14
        self.layer4 = ConvBNReLU(
            num_filters_init * 2, num_filters_init * 2, 3, 1, 1, bias=False
        )  # 14 -> 7

        self.layer5 = ConvBNReLU(
            num_filters_init * 2, num_filters_init * 4, 12, 1, 1, bias=False
        )  # 7 -> 3
        self.layer6 = ConvBNReLU(
            num_filters_init * 4, num_filters_init * 4, 12, 1, 1, bias=False
        )  # 6 -> 1

        self.layer7 = ConvBNReLU(
            num_filters_init * 4, num_filters_init * 8, 36, 1, 1, bias=False
        )  # 6 -> 1
        self.layer8 = ConvBNReLU(
            num_filters_init * 8, num_filters_init * 8, 36, 1, 1, bias=False
        )  # 7 -> 3

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())

        out = self.layer2(out)
        # print(out.size())
        out = self.max_pool(out)

        out = self.layer3(out)
        # print(out.size())

        out = self.layer4(out)
        out = self.max_pool(out)

        out = self.layer5(out)
        # print(out.size())
        out = self.layer6(out)
        out = self.max_pool(out)
        out = self.layer7(out)

        out = self.layer8(out)
        out = self.max_pool(out)

        out = torch.flatten(out, 1)
        return out


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_cnn_channels=3,
        num_cnn_filters_init=32,
        lstm_layer=3,
        lstm_nn_size=1024,
        model_device="cpu",
        dropout_p=0,
        bidrectional=False,
        batch_size=10,
    ):
        super(CNNLSTM, self).__init__()
        self.feature_extractor = CNN(
            in_channels=in_cnn_channels, num_filters_init=num_cnn_filters_init
        )
        self.lstm = nn.LSTM(
            input_size=4608,
            hidden_size=lstm_nn_size,
            num_layers=lstm_layer,
            bidirectional=bidrectional,
        )
        if bidrectional is True:
            fc_feature_size = lstm_nn_size * 2
        else:
            fc_feature_size = lstm_nn_size
        self.fc1 = nn.Linear(fc_feature_size, fc_feature_size)
        self.dropout_layer = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(fc_feature_size, num_classes)
        self.fc_feature_size = fc_feature_size
        self.model_device = model_device

        self.lstm_layer = lstm_layer
        self.batch_size = batch_size
        self.lstm_nn_size = lstm_nn_size
        self.bidrectional = bidrectional

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        init_lstm_layer = self.lstm_layer
        if self.bidrectional:
            init_lstm_layer = self.lstm_layer * 2
        hidden_a = torch.randn(
            init_lstm_layer,
            batch_size,
            self.lstm_nn_size,
            device=self.model_device,
        )
        hidden_b = torch.randn(
            init_lstm_layer,
            batch_size,
            self.lstm_nn_size,
            device=self.model_device,
        )

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, x, seq_lengths):
        # x dim: batch_size x C x F_1
        # we will need to do the packing of the sequence
        # dynamically for each batch of input
        # 1. feature extractor

        x = self.feature_extractor(x)  # x dim: total_epoch_num * feature size
        feature_size = x.size()[-1]

        # 2. lstm
        seq_tensor = torch.zeros(
            len(seq_lengths),
            seq_lengths.max(),
            feature_size,
            dtype=torch.float,
            device=self.model_device,
        )
        start_idx = 0
        for i in range(len(seq_lengths)):
            current_len = seq_lengths[i]
            current_series = x[
                start_idx : start_idx + current_len, :
            ]  # num_time_step x feature_size
            current_series = current_series.view(
                1, current_series.size()[0], -1
            )
            seq_tensor[i, :current_len, :] = current_series
            start_idx += current_len

        seq_lengths_ordered, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        packed_input = pack_padded_sequence(
            seq_tensor, seq_lengths_ordered.cpu().numpy(), batch_first=True
        )

        # x dim for lstm: #  batch_size_rnn x Sequence_length x F_2
        # uncomment for random init state
        # hidden = self.init_hidden(len(seq_lengths))
        packed_output, _ = self.lstm(packed_input)
        output, input_sizes = pad_packed_sequence(
            packed_output, batch_first=True
        )

        # reverse back to the original order
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = output[unperm_idx]

        # reverse back to the originaly shape
        # total_epoch_num * fc_feature_size
        fc_tensor = torch.zeros(
            seq_lengths.sum(),
            self.fc_feature_size,
            dtype=torch.float,
            device=self.model_device,
        )

        start_idx = 0
        for i in range(len(seq_lengths)):
            current_len = seq_lengths[i]
            current_series = lstm_output[
                i, :current_len, :
            ]  # num_time_step x feature_size
            current_series = current_series.view(current_len, -1)
            fc_tensor[start_idx : start_idx + current_len, :] = current_series
            start_idx += current_len
        # print("lstm time: ", end-start)

        # 3. linear readout
        x = self.fc1(fc_tensor)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [
        64,
        3,
        64,
        3,
        "M",
        128,
        3,
        128,
        3,
        128,
        3,
        "M",
        256,
        3,
        256,
        3,
        256,
        3,
        256,
        3,
        "M",
        512,
        3,
        512,
        3,
        512,
        3,
        512,
        3,
        "M",
        512,
        3,
        512,
        3,
        512,
        3,
        512,
        3,
        "M",
        1024,
        30,
    ],  # converted one FC to ConV
    "B": [
        64,
        12,
        64,
        12,
        "M",
        128,
        24,
        128,
        24,
        "M",
        256,
        24,
        256,
        24,
        "M",
        512,
        24,
        512,
        24,
        "M",
        512,
        24,
        512,
        48,
        "M",
    ],
    "C": [
        64,
        12,
        64,
        12,
        "M",
        128,
        24,
        128,
        24,
        "M",
        256,
        24,
        256,
        24,
        "M",
        512,
        48,
        512,
        48,
        "M",
        512,
        48,
        512,
        92,
        "M",
    ],
    "D": [32, 12, 64, 12, "M", 128, 24, 128, 48, "M", 256, 48, 256, 96, "M"],
}


def _cnn(
    cfg: str, batch_norm: bool, pretrained: bool, **kwargs: Any
) -> AccNet:
    if pretrained:
        kwargs["init_weights"] = False
    model = AccNet(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def cnn1(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 28  # this shouldn't change
    return _cnn(
        "A",
        True,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnn3(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 32  # this shouldn't change
    return _cnn(
        "B",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnn5(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
     Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 42  # this shouldn't change
    return _cnn(
        "C",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnnSmall(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 12  # this shouldn't change
    classifier_layer_size = 1000
    return _cnn(
        "D",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        classifier_layer_size=classifier_layer_size,
        **kwargs,
    )


class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        output_size=1,
        n_channels=3,
        is_eva=False,
        resnet_version=1,
        epoch_len=10,
        is_mtl=False,
        is_simclr=False,
    ):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor
        self.is_mtl = is_mtl

        # Classifier input size = last out_channels in previous layer
        if is_eva:
            self.classifier = EvaClassifier(
                input_size=out_channels, output_size=output_size
            )
        elif is_mtl:
            self.aot_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.scale_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.permute_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.time_w_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
        elif is_simclr:
            self.classifier = ProjectionHead(
                input_size=out_channels, encoding_size=output_size
            )

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)

        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h
        else:
            y = self.classifier(feats.view(x.shape[0], -1))
            return y
        return y


def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
