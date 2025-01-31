from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.ops import misc as misc_nn_ops
from .._utils import IntermediateLayerGetter
from .. import resnet, shufflenetv2


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list,
                 out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def resnet_fpn_backbone(backbone_name, pretrained):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3'
    }

    in_channels_stage2 = backbone.inplanes // 8
    print('backbone.inplanes = ', backbone.inplanes)
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list,
                           out_channels)


def shufflenet_fpn_backbone(backbone_name, pretrained):
    backbone = shufflenetv2.__dict__[backbone_name](
        pretrained=pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'stage3' not in name and 'stage4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'stage2': '0', 'stage3': '1', 'stage4': '2'}

    in_channels_stage2 = backbone._stage_out_channels[-2] // 4
    # print('backbone._stage_out_channels[1] = ',
    #       backbone._stage_out_channels[1])
    # print('backbone._stage_out_channels[-2] = ',
    #       backbone._stage_out_channels[-2])
    # print('backbone._stage_out_channels = ', backbone._stage_out_channels)
    in_channels_list = [
        in_channels_stage2, in_channels_stage2 * 2, in_channels_stage2 * 4
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list,
                           out_channels)
